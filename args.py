import argparse


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
    parser.add_argument('--g_online', action='store_true')
    parser.add_argument('--g_estim', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--gvar_iter',
                        default=argparse.SUPPRESS, type=int)
    args = parser.parse_args()
    return args
