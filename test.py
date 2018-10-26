from __future__ import print_function
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
from log_utils import AverageMeter
import time
from schedulers import class_sum
from autograd.nonacc import nonacc_grad


def test(tb_logger, model, gluster, test_loader,
         opt, niters, set_name='Test', prefix='V'):
    model.eval()
    if gluster:
        gluster.eval()
    # TODO: sma_eval needs optimizer
    test_loss = 0
    correct = 0
    test_loader.sampler.training = False
    loss_h = np.zeros(len(test_loader.dataset))
    targets = np.zeros(len(test_loader.dataset))
    num = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            num += len(idx)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target,
            #                         size_average=False).item()
            loss = F.nll_loss(output, target, reduction='none')
            test_loss += loss.sum().item()
            loss_h[idx] = loss.cpu().numpy()
            targets[idx] = target.data.cpu().numpy()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            # tb_logger.log_value('%stop1' % prefix,
            #                     float(prec1), data.size(0))
            # tb_logger.log_value('%stop5' % prefix,
            #                     float(prec5), data.size(0))
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
    # logging.info('num: %d' % num)
    test_loader.sampler.training = True
    tb_logger.log_hist('%sloss_h' % prefix, loss_h, step=niters,
                       log_scale=True)
    for i in range(min(10, int(targets.max()))):
        tb_logger.log_hist('%slossC%d_h' % (prefix, i),
                           loss_h[targets == i], step=niters,
                           log_scale=True)
    closs_h = class_sum(loss_h, targets)
    tb_logger.log_hist('%slossC_h' % prefix, closs_h, step=niters,
                       log_scale=True, bins=min(len(closs_h), 100))
    if gluster:
        np.set_printoptions(suppress=True)
        L = [c.pow(2).view(gluster.nclusters, -1).sum(1).cpu().numpy()
             for c in gluster.get_centers()]
        normC = np.sum(L, 0)
        print('normC:')
        print(normC)
        print('Reinit count: %s' % str(gluster.reinits.cpu().numpy()))
        print('Cluster size: %s' % str(gluster.cluster_size.cpu().numpy()))
        np.set_printoptions(suppress=False)

    return accuracy


def stats(tb_logger,
          model, test_loader, opt, optimizer, set_name='Test', prefix='V'):
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    test_loader.sampler.training = False
    loss_f = np.zeros(len(test_loader.dataset))
    normg_f = np.zeros(len(test_loader.dataset))
    ggbar_f = np.zeros(len(test_loader.dataset))
    num = 0
    for batch_idx, (data, target, idx) in enumerate(test_loader):
        num += len(idx)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.nll_loss(output, target, reduction='none')
        grads = nonacc_grad(model, loss)
        normg, ggbar = batch_stats(optimizer, grads)
        loss_f[idx] = loss.detach().cpu().numpy()
        normg_f[idx] = normg
        ggbar_f[idx] = ggbar
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % opt.log_interval == 0:
            logging.info(
                '{0} set: [{1}/{2}]\t'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                .format(set_name, batch_idx, len(test_loader),
                        batch_time=batch_time))
    logging.info('num: %d' % num)
    test_loader.sampler.training = True
    tb_logger.log_obj('%sloss_f' % prefix, loss_f)
    tb_logger.log_obj('%snormg_f' % prefix, normg_f)
    tb_logger.log_obj('%sggbar_f' % prefix, ggbar_f)


def batch_stats(optimizer, grads_group):
    param_groups = optimizer.param_groups
    state = optimizer.state
    assert len(grads_group) == 1
    normg = np.zeros(len(grads_group[0]))
    ggbar = np.zeros(len(grads_group[0]))
    for group, grads in zip(param_groups, grads_group):
        param_state = [state[p] for p in group['params']]
        if 'momentum_buffer' in param_state[0]:
            buf = [p['momentum_buffer'].view(-1) for p in param_state]
        else:
            buf = [p.data.view(-1) for p in group['params']]
        gg = torch.cat(buf)
        for i in range(len(grads)):
            gf = [g.data.view(-1) for g in grads[i]]
            gc = torch.cat(gf)
            normg[i] += gc.pow(2).sum().item()
            ggbar += abs(torch.dot(gc, gg)).item()
    normg = np.sqrt(normg)
    return normg, ggbar
