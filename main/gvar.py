from __future__ import print_function
import numpy as np
import logging
import yaml
import os
import glob
import copy
import sys

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing

import utils
import models
from data import get_loaders, InfiniteLoader, get_gluster_loader
from args import add_args, opt_to_gluster_kwargs
from gluster.gluster import GradientClusterBatch, GradientClusterOnline
from log_utils import TBXWrapper
from log_utils import Profiler
from log_utils import LogCollector
tb_logger = TBXWrapper()
torch.multiprocessing.set_sharing_strategy('file_system')


class GradientEstimator(object):
    def __init__(self, data_loader, opt, *args, **kwargs):
        self.opt = opt
        self.model = None
        self.data_loader = data_loader

    def snap_batch(self, model, niters):
        pass

    def update_sampler(self):
        pass

    def snap_online(self, model, niters):
        pass

    def grad(self, model_new, in_place=False):
        raise NotImplemented('grad not implemented')

    def get_Ege_var(self, model, gviter):
        # estimate grad mean and variance
        Ege = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad(model)
            for e, g in zip(Ege, ge):
                e += g

        for e in Ege:
            e /= gviter
        nw = sum([w.numel() for w in model.parameters()])
        var_e = 0
        Es = [torch.zeros_like(g) for g in model.parameters()]
        En = [torch.zeros_like(g) for g in model.parameters()]
        for i in range(gviter):
            ge = self.grad(model)
            # TODO: is variance important?
            v = sum([(gg-ee).pow(2).sum() for ee, gg in zip(Ege, ge)])
            for s, e, g, n in zip(Es, Ege, ge, En):
                s += g.pow(2)
                n += (e-g).pow(2)
            var_e += v/nw

        var_e /= gviter
        # Division by gviter cancels out in ss/nn
        snr_e = sum(
                [((ss+1e-10).log()-(nn+1e-10).log()).sum()
                    for ss, nn in zip(Es, En)])/nw
        return Ege, var_e, snr_e


class SGDEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SGDEstimator, self).__init__(*args, **kwargs)
        # many open files? torch.multiprocessing sharing file_system
        self.data_iter = iter(InfiniteLoader(self.data_loader))

    def grad(self, model, in_place=False):
        data = next(self.data_iter)

        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # print(loss)
        # import ipdb; ipdb.set_trace()
        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g


class SVRGEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGEstimator, self).__init__(*args, **kwargs)
        self.data_iter = iter(InfiniteLoader(self.data_loader))

    def snap_batch(self, model, niters):
        self.model = model = copy.deepcopy(model)
        self.mu = [torch.zeros_like(g) for g in model.parameters()]
        num = 0
        batch_time = Profiler()
        for batch_idx, (data, target, idx) in enumerate(self.data_loader):
            num += len(idx)
            data, target = data.cuda(), target.cuda()
            model.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            grad_params = torch.autograd.grad(loss, model.parameters())
            for m, g in zip(self.mu, grad_params):
                m += g
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                logging.info(
                        'SVRG Snap> [{0}/{1}]: {bt}'.format(
                            batch_idx, len(self.data_loader),
                            bt=str(batch_time)))
        for m in self.mu:
            m /= num

    def grad(self, model_new, in_place=False):
        data = next(self.data_iter)

        model_old = self.model
        data, target = data[0].cuda(), data[1].cuda()

        # old grad
        model_old.zero_grad()
        output = model_old(data)
        loss = F.nll_loss(output, target)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        model_new.zero_grad()
        output = model_new(data)
        loss = F.nll_loss(output, target)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        if in_place:
            for m, go, gn, p in zip(
                    self.mu, g_old, g_new, model_new.parameters()):
                p.grad.copy_(m-go+gn)
            return loss
        ge = [m-go+gn for m, go, gn in zip(self.mu, g_old, g_new)]
        return ge


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

        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='none')
        # multiply by the size of the cluster
        w = 1.*M*Nk/N
        loss = (loss*w).mean()
        # print(loss)
        # import ipdb; ipdb.set_trace()
        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g


class GlusterBatchEstimator(GlusterEstimator):
    def __init__(self, *args, **kwargs):
        super(GlusterBatchEstimator, self).__init__(*args, **kwargs)

    def snap_batch(self, model, niters):
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

    def snap_online(self, model, niters):
        opt = self.opt
        if self.gluster is None:
            self.gluster = GradientClusterOnline(
                    model, **opt_to_gluster_kwargs(opt))
            self.gluster.deactivate()
        # Online Gluster is only fully active here
        self.gluster.activate()
        data = next(self.data_iter)
        data, target = data[0].cuda(), data[1].cuda()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        ai, batch_dist, iv = self.gluster.em_step()
        # for ivi in iv:
        #     self.assign_i[self.assign_i == ivi] = -1
        #     ai[ai == ivi] = -1
        self.gluster.deactivate()

        tb_logger.log_vector('go_bd_i', batch_dist.cpu().numpy())
        total_dist = self.gluster.total_dist
        tb_logger.log_value('gb_td', total_dist.sum().item(), step=niters)

    def snap_batch(self, model, niters):
        raw_loader = self.raw_loader
        train_size = len(raw_loader.dataset)
        assign_i = -np.ones(train_size)

        self.gluster.eval()
        batch_time = Profiler()
        for batch_idx, (data, target, idx) in enumerate(raw_loader):
            data, target = data.cuda(), target.cuda()
            model.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
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


class MinVarianceGradient(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.model = model
        sgd = SGDEstimator(data_loader, opt)
        # gluster or SVRG
        if opt.g_estim == 'gluster':
            if opt.g_online:
                gest = GlusterOnlineEstimator(data_loader, opt)
            else:
                gest = GlusterBatchEstimator(data_loader, opt)
        elif opt.g_estim == 'svrg':
            gest = SVRGEstimator(data_loader, opt)
        elif opt.g_estim == 'sgd':
            gest = SGDEstimator(data_loader, opt)
        self.sgd = sgd
        self.gest = gest
        self.opt = opt
        self.tb_logger = tb_logger
        self.init_snapshot = False
        self.gest_used = False

    def snap_batch(self, model, niters):
        model.eval()
        # model.train() # TODO: SVRG might have trouble with dropout
        self.gest.snap_batch(model, niters)
        self.init_snapshot = True

    def snap_online(self, model, niters):
        model.eval()  # TODO: keep train
        self.gest.snap_online(model, niters)

    def log_var(self, model, niters):
        model.eval()
        if not self.init_snapshot:
            return ''
        gviter = self.opt.gvar_estim_iter
        Ege, var_e, snr_e = self.gest.get_Ege_var(model, gviter)
        # TODO: make this hyperparam
        Esgd, var_s, snr_s = self.sgd.get_Ege_var(model, gviter)
        bias = torch.mean(torch.cat(
            [(ee-gg).abs().flatten() for ee, gg in zip(Ege, Esgd)]))
        # print(np.sum([ee.pow(2).sum().item() for ee in Esgd]))
        # print(np.sum([gg.pow(2).sum().item() for gg in Ege]))
        tb_logger.log_value('grad_bias', float(bias), step=niters)
        tb_logger.log_value('est_var', float(var_e), step=niters)
        tb_logger.log_value('sgd_var', float(var_s), step=niters)
        tb_logger.log_value('est_snr', float(snr_e), step=niters)
        tb_logger.log_value('sgd_snr', float(snr_s), step=niters)
        sgd_x, est_x = ('', '[X]') if self.gest_used else ('[X]', '')
        return ('G Bias: %.8f\t'
                '%sSGD Var: %.8f\t %sEst Var: %.8f\t'
                'SGD SNR: %.8f\t Est SNR: %.8f' % (
                    bias, sgd_x, var_s, est_x, var_e, snr_s, snr_e))

    def grad(self, niters):
        model = self.model
        model.train()
        if not self.opt.g_optim or niters < self.opt.g_optim_start:
            self.gest_used = False
            return self.sgd.grad(model, in_place=True)
        self.gest_used = True
        return self.gest.grad(model, in_place=True)


def test(tb_logger, model, test_loader,
         opt, niters, set_name='Test', prefix='V'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='none')
            test_loss += loss.sum().item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        wrong = len(test_loader.dataset)-correct
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        error = 100. * wrong / len(test_loader.dataset)
        logging.info(
            '\n{0} set: Average loss: {1:.4f}'
            ', Accuracy: {2}/{3} ({4:.2f}%)'
            ', Error: {5}/{3} ({6:.2f}%)\n'.format(
                set_name, test_loss, correct, len(test_loader.dataset),
                accuracy, wrong, error))
        tb_logger.log_value('%sloss' % prefix, test_loss, step=niters)
        tb_logger.log_value('%scorrect' % prefix, correct, step=niters)
        tb_logger.log_value('%swrong' % prefix, wrong, step=niters)
        tb_logger.log_value('%sacc' % prefix, accuracy, step=niters)
        tb_logger.log_value('%serror' % prefix, error, step=niters)
    return accuracy


def train(tb_logger, epoch, train_loader, model, optimizer, opt, test_loader,
          save_checkpoint, train_test_loader, gvar):
    batch_time = Profiler(k=100)
    model.train()
    profiler = Profiler()
    epoch_iters = int(np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    optimizer.logger.reset()
    # for batch_idx, (data, target, idx) in enumerate(train_loader):
    for batch_idx in range(opt.epoch_iters):
        profiler.start()
        # sgd step
        optimizer.zero_grad()
        loss = gvar.grad(optimizer.niters)
        optimizer.step()
        optimizer.niters += 1
        profiler.toc('optim')
        # snapshot
        if optimizer.niters % opt.g_osnap_iter == 0:
            # Frequent snaps
            gvar.snap_online(model, optimizer.niters)
            profiler.toc('snap_online')
        if ((optimizer.niters-opt.gvar_start) % opt.g_bsnap_iter == 0
                and optimizer.niters >= opt.gvar_start):
            # Rare snaps
            logging.info('Batch Snapshot')
            gvar.snap_batch(model, optimizer.niters)
            profiler.toc('snap_batch')
            profiler.end()
            logging.info('%s' % str(profiler))
        profiler.end()

        batch_time.toc('Time')
        batch_time.end()
        niters = optimizer.niters

        # if True:
        if batch_idx % opt.log_interval == 0:
            gvar_log = ''
            prof_log = ''
            if batch_idx % opt.gvar_log_iter == 0:
                gvar_log = '\t'+gvar.log_var(model, niters)
            if opt.log_profiler:
                prof_log = '\t'+str(profiler)

            logging.info(
                'Epoch: [{0}][{1}/{2}]({niters})\t'
                'Loss: {loss:.6f}\t'
                '{batch_time}\t'
                '{opt_log}{gvar_log}{prof_log}'.format(
                    epoch, batch_idx, len(train_loader),
                    loss=loss.item(),
                    batch_time=str(batch_time),
                    opt_log=str(optimizer.logger),
                    gvar_log=gvar_log,
                    prof_log=prof_log,
                    niters=niters))
        if batch_idx % opt.tblog_interval == 0:
            tb_logger.log_value('epoch', epoch, step=niters)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.log_value('lr', lr, step=niters)
            tb_logger.log_value('niters', niters, step=niters)
            tb_logger.log_value('batch_idx', batch_idx, step=niters)
            # tb_logger.log_value('batch_time', batch_time.val,
            #                     step=niters)
            tb_logger.log_value('loss', loss, step=niters)
            optimizer.logger.tb_log(tb_logger, step=niters)
        if optimizer.niters % epoch_iters == 0:
            if opt.train_accuracy:
                test(tb_logger,
                     model, train_test_loader, opt, optimizer.niters,
                     'Train', 'T')
            prec1 = test(tb_logger,
                         model, test_loader, opt, optimizer.niters)
            save_checkpoint(model, float(prec1), opt, optimizer)
            tb_logger.save_log()


def main():
    args = add_args()
    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in od.items():
        opt[k] = v
    opt = utils.DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)
    logfname = os.path.join(opt.logger_name, 'log.txt')
    logging.basicConfig(
            filename=logfname,
            format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        # TODO: remove deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    train_loader, test_loader, train_test_loader = get_loaders(opt)

    if opt.epoch_iters == 0:
        opt.epoch_iters = int(
                np.ceil(1.*len(train_loader.dataset)/opt.batch_size))
    opt.maxiter = opt.epoch_iters * opt.epochs

    model = models.init_model(opt)
    model.criterion = F.nll_loss

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr, momentum=opt.momentum,
                              weight_decay=opt.weight_decay,
                              nesterov=opt.nesterov)
    elif opt.optim == 'adam':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.lr,
                              weight_decay=opt.weight_decay)
    optimizer.niters = 0
    optimizer.logger = LogCollector(opt)
    epoch = 0
    save_checkpoint = utils.SaveCheckpoint()

    # optionally resume from a checkpoint
    model_path = os.path.join(opt.run_dir, opt.ckpt_name)
    if opt.resume:
        resume = glob.glob(opt.resume)
        resume = resume[0] if len(resume) > 0 else opt.resume
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            best_prec1 = checkpoint['best_prec1']
            save_checkpoint.best_prec1 = best_prec1
            print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
                  .format(model_path, epoch, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    gvar = MinVarianceGradient(model, train_loader, opt, tb_logger)
    while optimizer.niters < opt.epochs*opt.epoch_iters:
        optimizer.epoch = epoch
        if isinstance(opt.lr_decay_epoch, str):
            utils.adjust_learning_rate_multi(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
        else:
            utils.adjust_learning_rate(
                    optimizer, optimizer.niters//opt.epoch_iters, opt)
        ecode = train(
                tb_logger,
                epoch, train_loader, model, optimizer, opt, test_loader,
                save_checkpoint, train_test_loader, gvar)
        if ecode == -1:
            break
        epoch += 1
    tb_logger.save_log()


if __name__ == '__main__':
    main()
