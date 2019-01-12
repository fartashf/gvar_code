from __future__ import print_function
import numpy as np
import logging

import torch
import torch.nn
import torch.multiprocessing

from data import InfiniteLoader, get_gluster_loader
from args import opt_to_gluster_kwargs
from gluster.gluster import GradientClusterBatch, GradientClusterOnline
from log_utils import Profiler
from .sgd import SGDEstimator


class GlusterEstimator(SGDEstimator):
    def __init__(self, *args, **kwargs):
        super(GlusterEstimator, self).__init__(*args, **kwargs)
        self.gluster = None
        self.raw_loader, self.data_loader, self.sampler = get_gluster_loader(
                self.data_loader, self.opt)
        # self.data_iter = iter(InfiniteLoader(self.data_loader))

    def update_sampler(self, assign_i, cluster_size):
        self.data_loader.sampler.set_assign_i(assign_i, cluster_size)
        self.data_iter = iter(InfiniteLoader(self.data_loader))

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        assign_i = self.sampler.assign_i
        cluster_size = self.sampler.cluster_size
        idx = data[2]
        ci = assign_i[idx].flatten()
        M = (cluster_size > 0).sum()
        Nk = cluster_size[ci].flatten()
        N = cluster_size.sum()

        loss_i = model.criterion(model, data, reduction='none')
        # multiply by the size of the cluster
        w = 1.*M*Nk/N
        loss = (loss_i*w).mean()
        # print(loss)
        # import ipdb; ipdb.set_trace()
        if not (loss < 1):
            print(loss_i)
            print((loss_i*w).sort())
            print(Nk[(loss_i*w).sort()[1]])
            import ipdb
            ipdb.set_trace()
        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g


class GlusterBatchEstimator(GlusterEstimator):
    def __init__(self, *args, **kwargs):
        super(GlusterBatchEstimator, self).__init__(*args, **kwargs)

    def snap_batch(self, model, niters):
        tb_logger = self.tb_logger
        # TODO: do we need this for batch?
        # self.model = copy.deepcopy(self.model)
        # model = self.model
        opt = self.opt
        if self.gluster is None:
            self.gluster = GradientClusterBatch(
                    model, **opt_to_gluster_kwargs(opt))
            self.gluster.deactivate()
        else:
            self.gluster.copy_(model)
        # Batch Gluster is only active here
        self.gluster.activate()
        opt = self.opt
        citers = opt.gb_citers
        # TODO: refactor this
        nclusters = self.gluster.nclusters
        for i in range(citers):
            stat = self.gluster.update_batch(
                    self.raw_loader, len(self.raw_loader.dataset),
                    ci=i, citers=citers)
            normC = self.gluster.get_center_norms()
            total_dist, assign_i, target_i, pred_i, loss_i, topk_i, GG_i = stat
            correct = topk_i[:, 0]
            ls = np.zeros(nclusters)
            acc = np.zeros(nclusters)
            cs = np.zeros(nclusters)
            for c in range(nclusters):
                idx, _ = np.where(assign_i == c)
                ls[c] = loss_i[idx].sum()/max(1, len(idx))
                acc[c] = (correct[idx]*100.).mean()
                cs[c] = len(idx)
            tb_logger.log_vector('gb_norm', normC)
            tb_logger.log_vector('gb_size', cs)
            tb_logger.log_vector('gb_loss', ls)
            tb_logger.log_vector('gb_acc', acc)
            tb_logger.log_vector('gb_td_i', total_dist)
        cluster_size = self.gluster.cluster_size
        self.gluster.print_stats()
        assign_i = stat[1]
        self.gluster.deactivate()
        tb_logger.log_value('gb_td', total_dist.sum().item(), step=niters)
        self.update_sampler(assign_i, cluster_size)


class GlusterOnlineEstimator(GlusterEstimator):
    def __init__(self, *args, **kwargs):
        super(GlusterOnlineEstimator, self).__init__(*args, **kwargs)
        self.raw_iter = iter(InfiniteLoader(self.raw_loader))
        self.init_assign = False

    def snap_online(self, model, niters):
        tb_logger = self.tb_logger
        opt = self.opt
        if self.gluster is None:
            self.gluster = GradientClusterOnline(
                    model, **opt_to_gluster_kwargs(opt))
            self.gluster.deactivate()
        # Online Gluster is only fully active here
        self.gluster.activate()
        # TODO: maybe use its own grad
        # self.grad(model, in_place=True)
        # TODO: using raw_iter variance goes up in the end
        # data = next(self.raw_iter)
        data = next(self.data_iter)  # bootstraping
        loss = model.criterion(model, data)  # no reweighting grads
        loss.backward()
        w = 1
        # TODO: reweight correction is not helpful
        if self.init_assign:
            cluster_size = self.sampler.cluster_size
            Nk = cluster_size
            N = cluster_size.sum()
            w = 1.*Nk/N
            # print(w.sum().item())
        ai, batch_dist, iv = self.gluster.em_step(w)  # TODO: reweight here?
        # for ivi in iv:
        #     self.assign_i[self.assign_i == ivi] = -1
        #     ai[ai == ivi] = -1
        self.gluster.deactivate()

        tb_logger.log_vector('go_bd_i', batch_dist.cpu().numpy())
        cluster_size = self.gluster.cluster_size
        total_dist = self.gluster.total_dist/cluster_size.sum().clamp(1)
        tb_logger.log_value('gb_td', total_dist.sum().item(), step=niters)
        # print(list(self.gluster.cluster_size.cpu().numpy().flatten()))
        # print(self.gluster.cluster_size.sum().item())
        tb_logger.log_value('gb_cs', cluster_size[0].item(), step=niters)
        reinits = self.gluster.reinits.sum().item()
        tb_logger.log_value('gb_reinits', reinits, step=niters)
        # print(self.gluster.cluster_size)

    def snap_batch(self, model, niters):
        raw_loader = self.raw_loader
        train_size = len(raw_loader.dataset)
        assign_i = -np.ones(train_size)

        self.gluster.eval()
        batch_time = Profiler()
        for batch_idx, data in enumerate(raw_loader):
            idx = data[2]
            loss = model.criterion(model, data)
            loss.backward()
            ai, batch_dist = self.gluster.em_step()
            assign_i[idx] = ai.cpu().numpy().flat
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                u, c = np.unique(assign_i, return_counts=True)
                logging.info(
                        'Assignment: [{0}/{1}]: {bt}\t{2}\t{3}'.format(
                            batch_idx, len(raw_loader),
                            str(list(u)), str(list(c)), bt=str(batch_time)))
        u, c = np.unique(assign_i, return_counts=True)
        self.gluster.deactivate()

        assign_i = np.array(assign_i, dtype='int')
        assert 0 <= assign_i.min() and assign_i.max() < self.gluster.nclusters
        cluster_size = torch.zeros_like(self.gluster.cluster_size)
        cluster_size[u, 0] = torch.from_numpy(c).cuda().float()
        self.update_sampler(assign_i, cluster_size)
        self.init_assign = True
        # self.gluster.reset()
