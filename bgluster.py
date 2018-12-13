from __future__ import print_function
import numpy as np
import logging
import yaml
import os
import time

import torch
import torch.nn
import torch.backends.cudnn as cudnn

import utils
import models
from data import get_loaders
from args import add_args
from gluster.gluster import GradientClusterBatch


def test(model, data_loader, opt, dataset):
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
            model, opt.g_min_size, nclusters=opt.g_ncluster,
            no_grad=opt.g_no_grad, active_only=opt.g_active_only)

    gluster_tc = np.zeros(citers)
    total_dist = float('inf')
    pred_i = 0
    loss_i = 0
    for i in range(citers):
        tic = time.time()
        stat = gluster.update_batch(
                data_loader, len(data_loader.dataset),
                ci=i, citers=citers)
        if i > 0:
            assert pred_i.sum() == stat[3].sum(), 'predictions changed'
            assert loss_i.sum() == stat[4].sum(), 'loss changed'
            assert stat[0] <= total_dist, 'Total distortions went up'
        # TODO: top5
        total_dist, assign_i, target_i, pred_i, loss_i, topk_i = stat
        toc = time.time()
        gluster_tc[i] = (toc - tic)
        normC = gluster.print_stats()
        logging.info('%.4f +/- %.4f' % (gluster_tc.mean(), gluster_tc.std()))
        torch.save({'assign': assign_i, 'target': target_i,
                    'pred': pred_i, 'loss': loss_i,
                    'topk': topk_i,
                    'normC': normC, 'gtime': gluster_tc,
                    'opt': opt.d, 'dataset': dataset},
                   gb_fname)


def main():
    args = add_args()
    yaml_path = os.path.join('options/{}/{}'.format(args.dataset,
                                                    args.path_opt))
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    od = vars(args)
    for k, v in od.iteritems():
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

    # test(model, data_loader, opt, 'train_test')
    test(model, data_loader, opt, 'test')


if __name__ == '__main__':
    main()
