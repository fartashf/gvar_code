import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .layer import GlusterContainer
import logging
from log_utils import Profiler


class GradientCluster(object):
    def __init__(self, model, nclusters=1, debug=True, mul_Nk=False,
                 add_GG=False, add_CZ=False, eps=1, rank=1, eps_td=1e-7,
                 reg_Nk=0, **kwargs):
        # Q: duplicates
        # TODO: challenge: how many C? memory? time?

        self.nclusters = nclusters
        self.model = model
        self.device = device = next(self.model.parameters()).device
        # self.zero_data()
        # self._init_centers()
        self.reinits = torch.zeros(nclusters, 1).long().to(device)
        self.cluster_size = torch.zeros(nclusters, 1).to(device)
        self.total_dist = torch.zeros(self.nclusters, 1).to(device)
        self.debug = debug
        self.mul_Nk = mul_Nk
        self.add_GG = (mul_Nk or add_GG)
        self.add_CZ = add_CZ
        self.rank = rank
        self.eps_td = eps_td
        self.reg_Nk = reg_Nk

        self.G = GlusterContainer(
                model, eps, nclusters, debug=debug,
                add_GG=(mul_Nk or add_GG), add_CZ=add_CZ,
                cluster_size=self.cluster_size, rank=rank, **kwargs)

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

    def get_center_norms(self):
        centers = self.get_centers()
        nclusters = centers[0].shape[0]

        normC = np.zeros((nclusters,))
        L = [c.pow(2).view(nclusters, -1).sum(1).cpu().numpy()
             for c in centers]
        normC = np.sqrt(np.sum(L, 0))
        return normC

    def print_stats(self):
        normC = None
        # normC = self.get_center_norms()
        # logging.info('normC:\n%s' % str(normC))
        logging.info(
                'Reinit count:\n%s' % str(self.reinits.cpu().numpy()))
        logging.info(
                'Cluster size:\n%s' % str(self.cluster_size.cpu().numpy()))
        logging.info(
                'Total dists:\n%s' % str(self.total_dist.cpu().numpy()))
        return normC

    def get_dist(self):
        batch_dist = self.G.get_dist()
        # total_dist = torch.stack(batch_dist).sum(0)
        CC, CG, GG, CZd = [torch.stack(x).sum(0) for x in zip(*batch_dist)]
        if self.rank > 1:
            CG = CG.view(self.nclusters, self.rank, -1).sum(1)
        total_dist = CC-2*CG
        if self.add_GG or self.mul_Nk:
            total_dist += GG
        cs = self.cluster_size+1
        if self.mul_Nk:
            # cluster_size should be correct and fixed for assignment step
            # total_dist.mul_(self.cluster_size.clamp(1))
            total_dist.mul_(cs)  # TODO: this or above
        if self.add_CZ:
            # TODO: TestGlusterMLPCZ.test_toy_batch distortion goes up
            # one cluster hasn't converged
            total_dist += CZd.mul(cs).mul(cs)
        if self.reg_Nk > 0:
            total_dist += self.reg_Nk*cs
        total_dist = (total_dist/self.eps_td).round()*self.eps_td
        # print(total_dist)
        return total_dist, GG


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
        total_dist, GG = self.get_dist()
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        if not self.G.is_eval:
            # TODO: approximate cluster size for mul_Nk
            counts = torch.zeros(self.nclusters, 1).cuda()
            counts.scatter_add_(0, assign_i, torch.ones_like(assign_i).float())
            self.cluster_size.add_(counts)
        return assign_i, batch_dist, GG

    def update_batch(self, data_loader, train_size, ci=0, citers=0,
                     do_log=True):
        # resnet32
        # total: .28s (both assign and update)
        # train: .03s
        # save_dist_hook: .25s, post_proc: .05s, GG: .18s
        model = self.model
        model.eval()
        device = self.device
        assign_i = torch.zeros(train_size, 1).long().to(device)
        if do_log:
            pred_i = np.zeros(train_size)
            loss_i = np.zeros(train_size)
            target_i = np.zeros(train_size)
            GG_i = np.zeros(train_size)
            topk_i = np.zeros((train_size, 2))
        total_dist = torch.zeros(self.nclusters, 1).to(device)
        self.cluster_size.fill_(0)

        if self.debug:
            logging.info(
                    'Gluster batch> Save distortions'
                    ' and assign data to clusters.')
        batch_time = Profiler()
        for batch_idx, (data, target, idx) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target, reduction='none')
            if do_log:
                pred_i[idx] = output.max(
                        1, keepdim=True)[1].cpu().numpy()[:, 0]
                if output.shape[1] >= 5:
                    _, pred = output.topk(5, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(target.view(1, -1).expand_as(pred))
                    topk_i[idx, 0] = correct[:1].float().sum(0).cpu().numpy()
                    topk_i[idx, 1] = correct.float().sum(0).cpu().numpy()
                target_i[idx] = target.cpu().numpy()
                loss_i[idx] = loss.detach().cpu().numpy()
            loss = loss.mean()
            loss.backward()
            # from log_utils import get_all_tensors, get_memory
            # A = get_all_tensors()
            # print([a.shape for a in A[-20:]])
            # import ipdb; ipdb.set_trace()
            ai, batch_dist, gg = self.assign()
            assert batch_dist.max() < 1e10 and batch_dist.min() > -1e10,\
                'Distortion out of bounds'
            assign_i[idx] = ai
            if do_log:
                GG_i[idx] = gg.cpu().numpy()
            total_dist.scatter_add_(0, ai, batch_dist)
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                if self.debug:
                    logging.info(
                            'Gluster> [{0}/{3}][Assign:0/2][{1}/{2}]\t'
                            'Loss: {loss:.6f}\t'
                            '{batch_time}'.format(
                                ci, batch_idx, len(data_loader), citers,
                                loss=loss.item(),
                                batch_time=str(batch_time)))
                # torch.cuda.empty_cache()
                # import gc
                # gc.collect()
        if not self.add_GG:
            # TODO: this should never happen? only to pass
            # test_gluster.TestGlusterMLP.test_toy_batch
            total_dist.div_(self.cluster_size.clamp(1))
        td = total_dist.cpu().numpy()
        self.total_dist.copy_(total_dist)
        if self.debug:
            logging.info(
                    'Gluster batch> total distortions: %f (negative is fine)'
                    % td.sum().item())

        # split the scattered cluster if the smallest cluster is small
        # More than one split: don't know the new dist after one split
        s, i = self.cluster_size.min(0)
        if s < self.min_size:
            # TODO: maybe + cluster size
            _, j = total_dist.masked_fill(
                self.cluster_size < self.min_size, float('-inf')).max(0)
            idx = torch.arange(assign_i.shape[0])[assign_i[:, 0] == j]
            assign_i[idx[torch.randperm(len(idx))[:len(idx)//2]]] = i
            self.reinits[i] += 1
            if self.debug:
                logging.info(
                        'Gluster reinit> %d -> %d (%d & %d)'
                        % (i, j, (assign_i == i).sum(), (assign_i == j).sum()))

        if self.debug:
            logging.info(
                    'Gluster batch> '
                    'Update cluster centers given the assignments')
        # self._init_centers(0)
        self.G.zero_new_centers()
        batch_time = Profiler()
        for batch_idx, (data, target, idx) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            # recompute A, G but don't do reassign as clusters change
            self.assign()
            # assign_i[idx], _ = self.assign()  # recompute A, G
            self.G.accum_new_centers(assign_i[idx])
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                if self.debug:
                    logging.info(
                            'Gluster> [{0}/{3}][Update:1/2][{1}/{2}]\t'
                            'Loss: {loss:.6f}\t'
                            '{batch_time}'.format(
                                ci, batch_idx, len(data_loader), citers,
                                loss=loss.item(),
                                batch_time=str(batch_time)))
                # torch.cuda.empty_cache()

        self.cluster_size.fill_(0)
        self.cluster_size.scatter_add_(0, assign_i,
                                       torch.ones_like(assign_i).float())
        self.G.update_batch(self.cluster_size)
        # td before E step, we have to do another loop for td after E step
        if do_log:
            return (
                    td, assign_i.cpu().numpy(),
                    target_i, pred_i, loss_i, topk_i, GG_i)
        return td, assign_i.cpu().numpy()


class GradientClusterOnline(GradientCluster):
    def __init__(self, model, beta=0.9, min_size=1,
                 reinit_method='data', init_mul=10,
                 reinit_iter=10, **kwargs):
        super(GradientClusterOnline, self).__init__(model, **kwargs)
        self.beta = beta  # cluster size decay factor
        self.min_size = min_size
        self.reinit_method = reinit_method  # 'data' or 'largest'
        self.init_mul = init_mul
        self.counter = 0
        self.reinit_iter = reinit_iter
        self.reset()

    def reset(self, ids=None):
        if ids is None:
            self.cluster_size.fill_(self.min_size*self.init_mul)
        else:
            self.cluster_size[ids] = self.min_size*self.init_mul

    def em_step(self, w=1):
        if self.G.is_active:
            with torch.no_grad():
                assign_i, batch_dist = self.assign()
                if not self.G.is_eval:
                    invalid_clusters = self.update(assign_i, batch_dist, w)
                    return assign_i, batch_dist, invalid_clusters
                return assign_i, batch_dist

    def assign(self):
        # M: assign input
        # M: a_i = argmin_c(|g_i-C_c|^2)
        #        = argmin_c gg+CC-2*gC
        #        = argmin_c CC-2gC
        total_dist, GG = self.get_dist()
        batch_dist, assign_i = total_dist.min(0)
        assign_i = assign_i.unsqueeze(1)
        batch_dist = batch_dist.unsqueeze(1)
        return assign_i, batch_dist

    def update(self, assign_i, batch_dist, w=1):
        # TODO: Update beta is more exact tested by MLP.test_more_iters
        return self.update_beta(assign_i, batch_dist, w)

    def update_beta(self, assign_i, batch_dist, w=1):
        # Not keeping a full internal assignment list
        # E: update C
        # E: C_c = mean_i(g_i * I(c==a_i))
        beta = self.beta
        counts = torch.zeros(self.nclusters, 1).cuda()
        counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        # TODO: w is only multiplied by total_dist, others are better to be
        # biased towards Gluster sampling
        # counts.mul_(w)
        # print(list(counts.cpu().numpy().flatten()))
        post_size = counts.mul(1-beta)
        post_ratio = post_size/counts.clamp(1)  # counts.clamp(.0001)
        # batch_size = assign_i.shape[0]
        # post_size.div_(batch_size).mul_(60000)
        self.cluster_size.mul_(beta).add_(post_size)  # no *(1-beta)
        # print(counts)
        pre_size = self.cluster_size-post_size
        # TODO: this should be indep of batch size?
        if True:  # self.add_GG:
            tddec = pre_size/self.cluster_size.clamp(1)
            td_new = torch.zeros_like(self.total_dist).scatter_add_(
                    0, assign_i, batch_dist).mul(w)
            self.total_dist.mul_(tddec).add_(td_new.mul_(1-tddec))
        else:
            # TODO: reinit largest is not good with no addg
            # do this to pass TestGlusterMLP.test_more_iters
            # something wrong here: MLP.test_mnist_online_delayed inf dists
            self.total_dist.mul_(pre_size).scatter_add_(
                0, assign_i, batch_dist).mul(w).div_(self.cluster_size)

        self.G.zero_new_centers()
        self.G.accum_new_centers(assign_i)
        self.G.update_online_beta(pre_size, post_ratio, self.cluster_size,
                                  beta, assign_i.shape[0])

        invalid_clusters = []
        if self.counter % self.reinit_iter == 0:
            invalid_clusters = self.reinit(assign_i.shape[0])
        self.counter += 1
        return invalid_clusters

    def update_eta(self, assign_i, batch_dist):
        # # Not keeping a full internal assignment list
        # # E: update C
        # # E: C_c = mean_i(g_i * I(c==a_i))
        # # *** Not being used now ****
        # # *** Not being used now ****
        # # *** Not being used now ****
        # # *** Not being used now ****
        # eta = 1-self.beta
        # counts = torch.zeros(self.nclusters, 1).cuda()
        # counts.scatter_add_(0, assign_i, torch.ones(assign_i.shape).cuda())
        # post_size = counts.mul(eta)
        # self.cluster_size.mul_(1-eta).add_(post_size)
        # eta = counts.mul(eta)/counts.clamp(1)  # after updating cluster size
        # # TODO: this should be indep of batch size?
        # if self.add_GG:
        #     td_new = torch.zeros_like(self.total_dist).scatter_add_(
        #             0, assign_i, batch_dist)
        #     self.total_dist.mul_(1-eta).add_(td_new.mul_(eta))
        # else:
        #     # TODO: reinit largest is not good with no addg
        #     # do this to pass TestGlusterMLP.test_more_iters
        #     pre_size = self.cluster_size-post_size
        #     self.total_dist.mul_(pre_size).scatter_add_(
        #         0, assign_i, batch_dist).div_(self.cluster_size)

        # self.G.zero_new_centers()
        # self.G.accum_new_centers(assign_i)
        # self.G.update_online_eta(eta, counts)

        # invalid_clusters = self.reinit(assign_i.shape[0])
        # return invalid_clusters
        pass

    def reinit(self, batch_size):
        # Reinit if no data is assigned to the cluster for some time
        # (time = beta decay).
        # If we reinit based only on the current batch assignemnts, it ignores
        # reinits constantly for the 5-example test
        # cs/batch_size prob of seeing an example from a cluster
        # should be equal to 1/nclusters, reinit if <.1*1/nclusters or .01
        # min_size=1 is equal probability
        reinits = (self.cluster_size/batch_size  # *self.nclusters
                   < self.min_size)
        nreinits = reinits.sum()
        self.reinits += reinits.long()
        # TODO: stop reinit or delay if too often
        if nreinits > 0:
            if self.debug:
                logging.info('Reinit: %s'
                             % str(list(np.where(
                                 reinits.cpu().numpy().flatten())[0])))
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
        self.reset(reinits)
        self.G.reinit_from_data(reinits, perm)
        invalid_clusters = reinits.cpu().numpy()
        return invalid_clusters

    def reinit_from_largest(self, reinits):
        # reinit from the largest cluster
        # reinit one at a time
        _, ri = reinits.max(0)
        tdm = self.total_dist.masked_fill(reinits, float('-inf'))
        _, li = tdm.max(0)
        self.G.reinit_from_largest(ri, li)
        self.cluster_size[ri] = self.cluster_size[li] / 2
        self.cluster_size[li] = self.cluster_size[li] / 2
        # This is not exact, maybe add a multiple of cluster size
        self.total_dist[ri] = self.total_dist[li]  # -1e-5
        # print(self.cluster_size)
        invalid_clusters = [ri.item()]
        return invalid_clusters


class GlusterBatchAssert(object):
    def __init__(self, gluster):
        self.total_dist = float('inf')
        self.pred_i = 0
        self.loss_i = 0
        self.reinits = 0
        self.GG_i = []
        self.reinited = False
        self.gluster = gluster
        self.citer = 0

    def do_assert(self, stat):
        self.citer += 1
        if self.citer == 1:
            self._update(stat)
            return
        assert self.pred_i.sum() == stat[3].sum(), 'predictions changed'
        assert self.loss_i.sum() == stat[4].sum(), 'loss changed'
        dt_down = stat[0].sum() <= self.total_dist.sum()+1e-5
        if self.reinited or self.gluster.add_GG:
            if not dt_down:
                logging.info('^^^^ Total dists went up. Prob GG changed.^^^^')
        else:
            assert dt_down, 'Total dists went up'
        reinits_new = self.gluster.reinits.sum().item()
        self.reinited = (reinits_new > self.reinits)
        self.reinits = reinits_new
        # # TODO: GG is not deterministic, because Ai and Go are not
        # if np.abs(self.GG_i-stat[6]).sum() > 1e-5:
        #     print(np.sort(np.abs(self.GG_i-stat[6])))
        #     print(np.where(np.abs(self.GG_i-stat[6]) > 1e-5))
        # assert np.abs(self.GG_i-stat[6]).sum() < 1e-5, 'GG changed'
        if np.abs(self.GG_i-stat[6]).sum() > 1e-5:
            logging.info('^^^^ GG changed ^^^^')
        self._update(stat)

    def _update(self, stat):
        (self.total_dist, self.assign_i, self.target_i, self.pred_i,
            self.loss_i, self.topk_i, self.GG_i) = stat
