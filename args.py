import argparse
import yaml
import os
from ast import literal_eval as make_tuple

import torch
import utils


def add_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
    parser.add_argument('--data', default=argparse.SUPPRESS,
                        type=str, help='path to data')
    parser.add_argument('--logger_name', default='runs/runX')
    parser.add_argument('--dataset', default='mnist', help='mnist|cifar10')

    # options that can be changed from default
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int, default=argparse.SUPPRESS, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=argparse.SUPPRESS,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='how many batches to wait before logging training'
                        ' status')
    parser.add_argument('--tblog_interval',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--optim', default=argparse.SUPPRESS, help='sgd|dmom')
    parser.add_argument('--dmom', type=float, default=argparse.SUPPRESS,
                        help='Data momentum')
    parser.add_argument('--low_theta', type=float, default=argparse.SUPPRESS,
                        help='Low threshold for discarding small norms.')
    parser.add_argument('--high_theta', type=float, default=argparse.SUPPRESS,
                        help='High threshold for discarding large norms.')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=argparse.SUPPRESS,
                        help='model architecture: (default: resnet32)')
    parser.add_argument('-j', '--workers', default=argparse.SUPPRESS,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--dmom_interval', default=argparse.SUPPRESS,
                        type=int,
                        metavar='W', help='Update dmom every X epochs.')
    parser.add_argument('--dmom_temp', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='Temperature [0, +inf),'
                        '0 is always dmom, +inf is never dmom')
    parser.add_argument('--alpha', default=argparse.SUPPRESS,
                        help='normg|jacobian')
    parser.add_argument('--jacobian_lambda', default=.5,
                        type=float)
    parser.add_argument('--dmom_theta', type=float, default=0,
                        help='max of dmom/normg')  # x/10, (x/10)^(2)
    parser.add_argument('--sampler', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--train_accuracy', action='store_true',
                        default=argparse.SUPPRESS)
    # parser.add_argument('--save_for_notebooks', action='store_true',
    #                     default=False)
    parser.add_argument('--alpha_norm', default=argparse.SUPPRESS,
                        help='sum_norm|exp_norm|nonorm')
    parser.add_argument('--sampler_weight', default=argparse.SUPPRESS,
                        help='dmom|alpha|normg')
    # parser.add_argument('--wmomentum', action='store_true',
    #                     default=argparse.SUPPRESS)
    parser.add_argument('--log_profiler', action='store_true')
    parser.add_argument('--log_image', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--lr_decay_epoch',
                        default=argparse.SUPPRESS)
    parser.add_argument('--norm_temp', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_th',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_alpha_perc',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sampler_w2c', default=argparse.SUPPRESS,
                        help='1_over|log')
    parser.add_argument('--sampler_max_count',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--sampler_start_epoch',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_update_interval',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_update_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--sampler_lr_update',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--sampler_lr_window',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--divlen',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--wmomentum', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--log_keys',
                        default='touch,touch_p,alpha_normed_h,count_h,tau')
    parser.add_argument('--sampler_params', default='10,90,5,100')
    parser.add_argument('--sampler_repetition',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--exp_lr',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--corrupt_perc',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--log_nex',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--lr_scale',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--sampler_maxw',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--data_aug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--wnoise_stddev',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--min_batches',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--tauXstd',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--tauXstdL',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--scheduler',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--pretrained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nodropout',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--num_class',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--minvar',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--instantw',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--lr_decay_rate',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--sma_momentum',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--nesterov',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--log_stats',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--label_smoothing',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--duplicate',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--alpha_diff',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--gluster',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_nclusters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_beta',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_min_size',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_reinit',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--gb_citers',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--run_dir', default='runs/runX')
    parser.add_argument('--ckpt_name', default='model_best.pth.tar')
    parser.add_argument('--g_no_grad', action='store_true')
    parser.add_argument('--g_active_only', default='')
    parser.add_argument('--g_inactive_mods', default='')
    parser.add_argument('--g_online', action='store_true')
    parser.add_argument('--g_estim', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--epoch_iters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_log_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_estim_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_debug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--gvar_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_noMulNk',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_bsnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_osnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_optim',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_optim_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_svd',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_CZ',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--no_transform',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_init_mul',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_reinit_iter',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_optim_max',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_reg_Nk',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_imbalance',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--half_trained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_resume',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_epoch',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_stable',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_avg',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_save_snap',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_noise',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_batch_size',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_msnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--adam_betas',
                        default=argparse.SUPPRESS, type=str)
    parser.add_argument('--adam_eps',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_mlr',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--g_whiten',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--svrg_bsnap_num',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--niters',
                        default=argparse.SUPPRESS, type=int)
    args = parser.parse_args()
    return args


def opt_to_gluster_kwargs(opt):
    active_only = (list(opt.g_active_only.split(','))
                   if opt.g_active_only != '' else [])
    inactive_mods = (list(opt.g_inactive_mods.split(','))
                     if opt.g_inactive_mods != '' else [])
    return {'beta': opt.g_beta, 'min_size': opt.g_min_size,
            'nclusters': opt.g_nclusters, 'reinit_method': opt.g_reinit,
            'no_grad': opt.g_no_grad, 'active_only': active_only,
            'inactive_mods': inactive_mods,
            'debug': opt.g_debug, 'mul_Nk': (not opt.g_noMulNk),
            'do_svd': opt.g_svd, 'add_CZ': opt.g_CZ,
            'init_mul': opt.g_init_mul*opt.batch_size,
            'reinit_iter': opt.g_reinit_iter,
            'reg_Nk': opt.g_reg_Nk, 'stable': opt.g_stable,
            'gnoise': opt.g_noise,
            'adam_betas': opt.adam_betas, 'adam_eps': opt.adam_eps,
            'do_whiten': opt.g_whiten}


def yaml_opt(yaml_path):
    opt = {}
    with open(yaml_path, 'r') as handle:
        opt = yaml.load(handle)
    return opt


def get_opt():
    args = add_args()
    opt = yaml_opt('options/default.yaml')
    opt_s = yaml_opt(os.path.join('options/{}/{}'.format(args.dataset,
                                                         args.path_opt)))
    opt.update(opt_s)
    opt.update(vars(args).items())
    # od = vars(args)
    # for k, v in od.items():
    #     opt[k] = v
    opt = utils.DictWrapper(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    if opt.g_batch_size == -1:
        opt.g_batch_size = opt.batch_size
    opt.adam_betas = make_tuple(opt.adam_betas)
    return opt
