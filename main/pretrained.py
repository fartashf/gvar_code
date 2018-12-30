from __future__ import print_function
import numpy as np
import logging
import yaml
import os
import time

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
import models
from data import get_loaders
from args import add_args
from gluster.gluster import GradientClusterBatch, GradientClusterOnline
from log_utils import AverageMeter


def test_batch(model, data_loader, opt, dataset):
    citers = opt.gb_citers

    try:
        os.makedirs(opt.run_dir)
    except os.error:
        pass
    gb_fname = os.path.join(opt.run_dir, 'bgluster_%s.pth.tar' % dataset)

    # data = train_loader, test_loader, train_test_loader
    if dataset == 'train_test':
        data_loader = data_loader[2]
    elif dataset == 'test':
        data_loader = data_loader[1]

    # model's weight are not going to change, opt.step() is not called
    gluster = GradientClusterBatch(
            model, opt.g_min_size, nclusters=opt.g_nclusters,
            no_grad=opt.g_no_grad, active_only=opt.g_active_only)

    gluster_tc = np.zeros(citers)
    total_dist = []
    pred_i = 0
    loss_i = 0
    reinits = 0
    reinited = False
    for i in range(citers):
        tic = time.time()
        stat = gluster.update_batch(
                data_loader, len(data_loader.dataset),
                ci=i, citers=citers)
        toc = time.time()
        gluster_tc[i] = (toc - tic)
        normC = gluster.print_stats()
        logging.info('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
        if i > 0:
            assert pred_i.sum() == stat[3].sum(), 'predictions changed'
            assert loss_i.sum() == stat[4].sum(), 'loss changed'
            dt_down = stat[0].sum() <= total_dist.sum()+1e-5
            if reinited:
                if not dt_down:
                    logging.info('^^^^ Total dists went up ^^^^')
            else:
                assert dt_down, 'Total dists went up'
        reinits_new = gluster.reinits.sum().item()
        reinited = (reinits_new > reinits)
        reinits = reinits_new
        total_dist, assign_i, target_i, pred_i, loss_i, topk_i = stat
        torch.save({'assign': assign_i, 'target': target_i,
                    'pred': pred_i, 'loss': loss_i,
                    'topk': topk_i,
                    'normC': normC, 'gtime': gluster_tc,
                    'opt': opt.d, 'dataset': dataset,
                    'total_dist': total_dist},
                   gb_fname)


def test_online(model, data_loader, opt, dataset):
    citers = opt.gb_citers

    try:
        os.makedirs(opt.run_dir)
    except os.error:
        pass
    gb_fname = os.path.join(opt.run_dir, 'bgluster_%s.pth.tar' % dataset)

    # data = train_loader, test_loader, train_test_loader
    if dataset == 'train_test':
        data_loader = data_loader[2]
    elif dataset == 'test':
        data_loader = data_loader[1]

    # model's weight are not going to change, opt.step() is not called
    gluster = GradientClusterOnline(
            model, opt.g_beta, opt.g_min_size,
            opt.g_reinit, nclusters=opt.g_nclusters,
            no_grad=opt.g_no_grad, active_only=opt.g_active_only)

    train_size = len(data_loader.dataset)
    gluster_tc = []
    assign_i = -np.ones((train_size, 1))
    pred_i = np.zeros(train_size)
    loss_i = np.zeros(train_size)
    target_i = np.zeros(train_size)
    topk_i = np.zeros((train_size, 2))
    ci = 0
    batch_time = AverageMeter()
    end = time.time()
    while ci < citers:
        for batch_idx, (data, target, idx) in enumerate(data_loader):
            tic = time.time()
            # model.train() # TODO: loss becomes 7. with this
            model.zero_grad()
            # TODO: optim
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # store stats
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
            ai, batch_dist, iv = gluster.em_step()
            ai = ai.cpu().numpy()
            assign_i[idx] = ai
            for ivi in iv:
                assign_i[assign_i == ivi] = -1
                ai[ai == ivi] = -1
            toc = time.time()
            gluster_tc += [toc - tic]
            batch_time.update(time.time() - end)
            end = time.time()
            ci += 1
            if ci >= citers:
                break
            if batch_idx % 10 == 0:
                logging.info(
                        'Epoch: [{0}/{1}]\t Loss: {loss:.6f}\t'
                        'Time: {batch_time.val: .3f}'
                        '({batch_time.avg:.3f})'.format(
                            ci, citers, loss=loss.item(),
                            batch_time=batch_time))
            if batch_idx % 50 == 0:
                logging.info('assign:')
                logging.info(ai[:10])
                logging.info(batch_dist[:10])
                normC = gluster.print_stats()
                logging.info('%.4f +/- %.4f' % (
                    np.mean(gluster_tc),
                    np.std(gluster_tc)))
                # import ipdb; ipdb.set_trace()
                u, c = np.unique(assign_i, return_counts=True)
                logging.info(u)
                logging.info(c)
                torch.save({'assign': assign_i, 'target': target_i,
                            'pred': pred_i, 'loss': loss_i,
                            'topk': topk_i,
                            'normC': normC, 'gtime': gluster_tc,
                            'opt': opt.d, 'dataset': dataset,
                            'total_dist': gluster.total_dist.cpu().numpy()},
                           gb_fname)

    u, c = np.unique(assign_i, return_counts=True)
    logging.info(u)
    logging.info(c)
    torch.save({'assign': assign_i, 'target': target_i,
                'pred': pred_i, 'loss': loss_i,
                'topk': topk_i,
                'normC': normC, 'gtime': gluster_tc,
                'opt': opt.d, 'dataset': dataset,
                'total_dist': gluster.total_dist.cpu().numpy()},
               gb_fname)


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

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    logging.info(str(opt.d))

    torch.manual_seed(opt.seed)
    if opt.cuda:
        # TODO: remove deterministic
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
    # helps with wide-resnet by reducing memory and time 2x
    cudnn.benchmark = True

    data_loader = get_loaders(opt)

    model = models.init_model(opt)
    # Deterministic predictions given dropout
    model.eval()

    # optionally resume from a checkpoint
    model_path = os.path.join(opt.run_dir, opt.ckpt_name)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        best_prec1 = checkpoint['best_prec1']
        print("=> loaded checkpoint '{}' (epoch {}, best_prec {})"
              .format(model_path, epoch, best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    if opt.g_online:
        test = test_online
    else:
        test = test_batch

    # test(model, data_loader, opt, 'train_test')
    test(model, data_loader, opt, 'test')


if __name__ == '__main__':
    main()
