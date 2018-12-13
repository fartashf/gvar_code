import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .layer import GlusterContainer
import logging
from log_utils import AverageMeter
import time


class GradientCluster(object):
    def __init__(self, model, nclusters=1, **kwargs):
        # Q: duplicates
        # TODO: challenge: how many C? memory? time?

        self.nclusters = nclusters
        self.model = model
        self.device = device = self.model.parameters().next().device
        # self.zero_data()
        # self._init_centers()
        self.reinits = torch.zeros(nclusters, 1).long().to(device)
        self.cluster_size = torch.zeros(nclusters, 1).to(device)
        self.total_dist = torch.zeros(self.nclusters, 1).to(device)

        self.G = GlusterContainer(model, 1, nclusters, **kwargs)

    def activate(self):
        self.G.activate()

    def eval(self):
        self.G.eval()

    def deactivate(self):
        self.G.deactivate()

    def copy_(self, model):
        for m, g in zip(model.parameters(), self.model.parameters()):
            g.data.copy_(m.data)

    def assign(self):
        raise NotImplemented('assign not implemented')

    def update(self, assign_i):
        raise NotImplemented('update not implemented')

    def get_centers(self):
        return self.G.get_centers()

    def print_stats(self):
        centers = self.get_centers()
        nclusters = centers[0].shape[0]

        normC = np.zeros((nclusters,))
        L = [c.pow(2).view(nclusters, -1).sum(1).cpu().numpy()
             for c in centers]
        normC = np.sqrt(np.sum(L, 0))
        logging.info('normC:')
        logging.info(normC)
        logging.info('Reinit count: %s' % str(self.reinits.cpu().numpy()))
        logging.info('Cluster size: %s' % str(self.cluster_size.cpu().numpy()))
        logging.info('Total dists: %s' % str(self.total_dist.cpu().numpy()))
        return normC


class GradientClusterBatch(GradientCluster):
    def __init__(self, model, min_size, **kwargs):
        super(GradientClusterBatch, self).__init__(model, **kwargs)
        self.min_size = min_size

    def assign(self):
        """Assign inputs to clusters
        M: assign input
        M: a_i = argmin_c(|g_i-C_c|^2)
        = argmin_c gg+CC-2*gC
        = argmin_c CC-2gC
        """
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        if not self.G.is_eval:
            counts = torch.zeros(self.nclusters, 1).cuda()
            counts.scatter_add_(0, assign_i, torch.ones_like(assign_i).float())
            self.cluster_size.add_(counts)
        return assign_i, batch_dist

    def update_batch(self, data_loader, train_size, ci=0, citers=0):
        model = self.model
        device = self.device
        assign_i = torch.zeros(train_size, 1).long().to(device)
        pred_i = np.zeros(train_size)
        loss_i = np.zeros(train_size)
        target_i = np.zeros(train_size)
        topk_i = np.zeros((train_size, 2))
        total_dist = torch.zeros(self.nclusters, 1).to(device)
        self.cluster_size = torch.zeros(self.nclusters, 1).cuda()

        logging.info(
                'Gluster batch> Save distortions'
                ' and assign data to clusters.')
        batch_time = AverageMeter()
        end = time.time()
        for batch_idx, (data, target, idx) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            # self.zero_data()
            output = self.model(data)
            pred_i[idx] = output.max(1, keepdim=True)[1].cpu().numpy()[:, 0]
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            topk_i[idx, 0] = correct[:1].float().sum(0).cpu().numpy()
            topk_i[idx, 1] = correct.float().sum(0).cpu().numpy()
            target_i[idx] = target.cpu().numpy()
            loss = F.nll_loss(output, target, reduction='none')
            loss_i[idx] = loss.detach().cpu().numpy()
            loss = loss.mean()
            loss.backward()
            ai, batch_dist = self.assign()
            assign_i[idx] = ai
            total_dist.scatter_add_(0, ai, batch_dist)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 10 == 0:
                logging.info(
                        'Gluster> [{0}/{3}][Assign:0/2][{1}/{2}]\t'
                        'Time: {batch_time.val: .3f}'
                        '({batch_time.avg:.3f})'.format(
                            ci, batch_idx, len(data_loader), citers,
                            batch_time=batch_time))
        total_dist.div_(self.cluster_size.clamp(1))
        td = total_dist.sum().item()
        self.total_dist.copy_(total_dist)
        logging.info(
                'Gluster batch> total distortions: %f (negative is fine)' % td)

        # split the scattered cluster if the smallest cluster is small
        # More than one split: don't know the new dist after one split
        s, i = self.cluster_size.min(0)
        if s < self.min_size:
            # TODO: negative distortion
            _, j = total_dist.masked_fill(
                self.cluster_size < self.min_size, float('-inf')).max(0)
            idx = torch.arange(assign_i.shape[0])[assign_i[:, 0] == j]
            assign_i[idx[torch.randperm(len(idx))[:len(idx)/2]]] = i
            self.reinits[i] += 1

        logging.info(
                'Gluster batch> Update cluster centers given the assignments')
        # self._init_centers(0)
        self.G.zero_new_centers()
        batch_time = AverageMeter()
        end = time.time()
        for batch_idx, (data, target, idx) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            # self.zero_data()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            # recompute A, G but don't do reassign as clusters change
            self.assign()
            # assign_i[idx], _ = self.assign()  # recompute A, G
            self.G.accum_new_centers(assign_i[idx])
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 10 == 0:
                logging.info(
                        'Gluster> [{0}/{3}][Update:1/2][{1}/{2}]\t'
                        'Time: {batch_time.val: .3f}'
                        '({batch_time.avg:.3f})'.format(
                            ci, batch_idx, len(data_loader), citers,
                            batch_time=batch_time))

        self.cluster_size.fill_(0)
        self.cluster_size.scatter_add_(0, assign_i,
                                       torch.ones_like(assign_i).float())
        self.G.update_batch(self.cluster_size)
        # td before E step, we have to do another loop for td after E step
        return td, assign_i.cpu().numpy(), target_i, pred_i, loss_i, topk_i


class GradientClusterOnline(GradientCluster):
    def __init__(self, model, beta=0.9, min_size=1,
                 reinit_method='data', **kwargs):
        super(GradientClusterOnline, self).__init__(model, **kwargs)
        self.beta = beta  # cluster size decay factor
        self.min_size = 1
        self.reinit_method = reinit_method  # 'data' or 'largest'
        self.cluster_size.fill_(self.min_size)

    def em_step(self):
        if self.G.is_active:
            with torch.no_grad():
                assign_i, batch_dist = self.assign()
                if not self.G.is_eval:
                    invalid_clusters = self.update(assign_i, batch_dist)
                    return assign_i, batch_dist, invalid_clusters
                return assign_i, batch_dist

    def assign(self):
        # M: assign input
        # M: a_i = argmin_c(|g_i-C_c|^2)
        #        = argmin_c gg+CC-2*gC
        #        = argmin_c CC-2gC
        batch_dist = self.G.get_dist()
        total_dist = torch.stack(batch_dist).sum(0)
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        return assign_i, batch_dist

    def update(self, assign_i, batch_dist):
        # Not keeping a full internal assignment list
        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        beta = self.beta
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        self.cluster_size.mul_(beta).add_(counts)  # no *(1-beta)
        pre_size = self.cluster_size-counts
        self.total_dist.mul_(pre_size).scatter_add_(
            0, assign_i, batch_dist).div_(self.cluster_size)

        self.G.zero_new_centers()
        self.G.accum_new_centers(assign_i)
        self.G.update_online(pre_size, self.cluster_size)

        invalid_clusters = self.reinit(assign_i.shape[0])
        return invalid_clusters

    def reinit(self, batch_size):
        # Reinit if no data is assigned to the cluster for some time
        # (time = beta decay).
        # If we reinit based only on the current batch assignemnts, it ignores
        # reinits constantly for the 5-example test
        reinits = (self.cluster_size < self.min_size)
        nreinits = reinits.sum()
        self.reinits += reinits.long()
        # TODO: stop reinit or delay if too often
        if nreinits > 0:
            if self.reinit_method == 'data':
                return self.reinit_from_data(reinits, batch_size)
            elif self.reinit_method == 'largest':
                return self.reinit_from_largest(reinits)
            else:
                raise Exception('Reinit method not defined.')
        return []

    def reinit_from_data(self, reinits, batch_size):
        nreinits = reinits.sum()
        # reinit from data
        perm = torch.randperm(batch_size)[:nreinits]
        self.cluster_size[reinits] = 1
        self.G.reinit_from_data(reinits, perm)
        invalid_clusters = reinits.cpu().numpy()
        return invalid_clusters

    def reinit_from_largest(self, reinits):
        # reinit from the largest cluster
        # reinit one at a time
        _, ri = reinits.max(0)
        tdm = self.total_dist.masked_fill(
            self.cluster_size < self.min_size, float('-inf'))
        # TODO: Ties should be broken deterministically
        _, li = tdm.max(0)
        self.G.reinit_from_largest(ri, li)
        self.cluster_size[ri] = self.cluster_size[li] / 2
        self.cluster_size[li] = self.cluster_size[li] / 2
        # This is not exact, maybe add a multiple of cluster size
        self.total_dist[ri] = self.total_dist[li]  # -1e-5
        # print(self.cluster_size)
        invalid_clusters = [ri]
        return invalid_clusters
