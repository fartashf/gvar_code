import numpy as np
import os
import re
import requests
import torch
import pylab as plt
import matplotlib.ticker as mtick
from data import get_loaders
import torchvision.utils as vutils
from torchvision import transforms
import utils
from collections import OrderedDict
import copy


class TBMultiplexer:

    def __init__(self, logdir):
        self.logdir = logdir
        SIZE_GUIDANCE = {'scalars': 1000}
        # using event multiplexer from tensorboard
        from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # NOQA
        self.multiplexer = event_multiplexer.EventMultiplexer(
            tensor_size_guidance=SIZE_GUIDANCE)
        self.multiplexer.AddRunsFromDirectory(logdir)

    def add_runs(self, patterns):
        run_names = get_run_names(self.logdir, patterns)
        for run_name in run_names:
            self.multiplexer.AddRun(run_name)

    def reload(self):
        self.multiplexer.Reload()

    def get_data(self, logdir, run_names, tag_names):
        data = []
        for run_name in run_names:
            d = {}
            import tensorboard_extract as tb_extract
            for tag_name in tag_names:
                js = tb_extract.extract_scalars(self.multiplexer, run_name[
                                                len(self.logdir):], tag_name)
                d[tag_name] = np.array([[x[j] for x in js]
                                        for j in range(1, 3)])
            data += [d]
        return data


def get_run_names(logdir, patterns):
    run_names = []
    for pattern in patterns:
        for root, subdirs, files in os.walk(logdir, followlinks=True):
            if re.match(pattern, root):
                run_names += [root]
    # print(run_names)
    run_names.sort()
    return run_names


def get_data(logdir, run_names, tag_names):
    data = []
    for run_name in run_names:
        d = {}
        for tag_name in tag_names:
            resp = requests.get(
                'http://localhost:6007/data/plugin/scalars/scalars?',
                params={'run': run_name[len(logdir):], 'tag': tag_name})
            # print(resp.url)
            js = resp.json()
            d[tag_name] = np.array([[x[j] for x in js] for j in range(1, 3)])
        data += [d]
    return data


def get_data_pth(logdir, run_names, tag_names, batch_size=None):
    data = []
    for run_name in run_names:
        d = {}
        if not os.path.exists(run_name + '/log.pth.tar'):
            logdata = {}
            logdata['gb_td'] = [(0, 1000, np.nan)]
        else:
            logdata = torch.load(run_name + '/log.pth.tar')
        if 'classes' in logdata:
            d['classes'] = logdata['classes']
        for tag_name in tag_names:
            tag_name_orig_nex = None
            if 'Nex' in tag_name and tag_name not in logdata:
                tag_name_orig_nex = tag_name
                tag_name = tag_name[:-4]
            tag_name_orig_log = None
            if 'nolog' in tag_name:
                idx = tag_name.find('nolog')
                tag_name_orig_log = tag_name[:idx-1]+tag_name[idx+5:]
                tag_name = tag_name[:idx]+tag_name[idx+2:]
            if '_h_' in tag_name and tag_name[-1].isdigit():
                tg = tag_name[:tag_name.rfind('_')]
                if tg not in logdata:
                    continue
                js = logdata[tg]
                h_index = int(tag_name[tag_name.rfind('_')+1:])
                niters = np.array([x[1] for x in js])
                h_iters = int(1. * h_index * niters[-1] / 100.)
                h_index = np.abs(niters-h_iters).argmin()
                logdata[tag_name] = [logdata[tg][h_index]]
            if '_log' in tag_name and tag_name not in logdata:
                tg = tag_name[:tag_name.find('_log')]
                if tg in logdata:
                    val = np.log(np.maximum(np.exp(-20), logdata[tg]))
                    logdata[tag_name] = val
            if tag_name not in logdata:
                continue
            js = logdata[tag_name]
            if '_h' in tag_name:
                d[tag_name] = js[-1][2]
                if tag_name_orig_log is not None:
                    d[tag_name_orig_log] = (d[tag_name][0],
                                            np.exp(d[tag_name][1]))
            elif '_f' in tag_name:
                js[np.isnan(js)] = 100
                js[np.isinf(js)] = 100
                d[tag_name] = np.histogram(js, bins=100)
            elif tag_name.endswith('_v'):
                d[tag_name] = (
                        np.array([x[1] for x in js]),
                        np.array([x[2] for x in js]))
            else:
                d[tag_name] = np.array([[x[j] for x in js]
                                        for j in range(1, 3)])
                if tag_name_orig_nex is not None:
                    d[tag_name_orig_nex] = np.array(d[tag_name])
                    d[tag_name_orig_nex][0] = d[tag_name][0] * batch_size
        data += [d]
    return data


def smooth_data(x, y, npts=300, order=3, s=0):
    x_smooth = np.linspace(x.min(), x.max(), npts)

    ftype = 5
    if ftype == 1:
        from scipy.interpolate import spline
        y_smooth = spline(x, y, x_smooth, order=order)
    elif ftype == 2:
        pass
        # from scipy.interpolate import make_interp_spline, BSpline
        # spl = make_interp_spline(x, y, k=order)
    elif ftype == 3:
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind='cubic')
        y_smooth = f(x_smooth)
    elif ftype == 4:
        from scipy.interpolate import UnivariateSpline
        f = UnivariateSpline(x, y, s=s, k=order)
        y_smooth = f(x_smooth)
    elif ftype == 5:
        if s == 1:
            return x, y
        from scipy.interpolate import LSQUnivariateSpline
        # t = np.linspace(-3, 3, nknots+2)[1:-1]
        spl = LSQUnivariateSpline(x, y, x_smooth[1:-1])
        y_smooth = spl(x_smooth)
    return x_smooth, y_smooth


def plot_smooth(x, y, npts=300, order=3, s=0, *args, **kwargs):
    x_smooth, y_smooth = smooth_data(x, y, npts, order, s)
    # x_smooth = x
    # y_smooth = y
    plt.plot(x_smooth, y_smooth, *args, **kwargs)
    return x_smooth, y_smooth


def rolling_window(a, window):
    # https://stackoverflow.com/a/59322185
    # pad = np.ones(len(a.shape), dtype=np.int32)
    # pad[-1] = window-1
    # pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    if window % 2 == 0:
        window += 1
    pad = int(window/2)
    a_pad = np.pad(a, (pad, pad), mode='reflect')
    # shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    # strides = a.strides + (a.strides[-1],)
    shape = (a.shape[0], window)
    strides = a_pad.strides + (a_pad.strides[-1],)
    return np.lib.stride_tricks.as_strided(a_pad, shape=shape, strides=strides)


def plot_smooth_shade(x, y, *args, s=None, log_scale=False, **kwargs):
    if s is None:
        s = 10
    npts = int(len(x)//s)
    order = 1
    # s = 0.1   # s=None smoothest, s=0 through all points
    x_smooth, y_smooth = plot_smooth(x, y, npts, order, s, *args, **kwargs)
    ystd = np.std(rolling_window(y, window=int(s*2)), axis=-1)
    # ystd = [ystd[int(x)] for x in np.linspace(0, len(x)-1, npts)]
    _, ystd = smooth_data(x, ystd, npts, order, s)
    # ystd = np.std(y)
    if not log_scale:
        plt.fill_between(x_smooth, y_smooth-ystd, y_smooth+ystd, alpha=0.2,
                         *args, **kwargs)
    else:
        yerr = np.exp(np.log(y_smooth)
                      - (np.log(y_smooth+ystd)-np.log(y_smooth)))
        plt.fill_between(x_smooth, yerr, y_smooth+ystd, alpha=0.1,
                         *args, **kwargs)


def plot_smooth_o1(x, y, *args, **kwargs):
    plot_smooth(x, y, 100, 1, 0, *args, **kwargs)


def plot_smooth_o2(x, y, *args, **kwargs):
    plot_smooth(x, y, 100, 2, 0.001, *args, **kwargs)


def plot_smooth_o5(x, y, *args, **kwargs):
    plot_smooth(x, y, 100, 5, 0.001, *args, **kwargs)


def get_legend(lg_tags, run_name, lg_replace=[]):
    lg = ""
    for lgt in lg_tags:
        res = ".*?($|,)" if ',' not in lgt and '$' not in lgt else ''
        mg = re.search(lgt + res, run_name)
        if mg:
            lg += mg.group(0)
    lg = lg.replace('_,', ',')
    lg = lg.strip(',')
    for a, b in lg_replace:
        lg = lg.replace(a, b)
    return lg


def plot_tag(data, plot_f, run_names, tag_name, lg_tags, ylim=None, color0=0,
             ncolor=None, lg_replace=[], no_title=False, smooth_div_n=None):
    xlabel = {'visits_h': '# Visits',
              'count_h': '# Wait Iterations',
              'Tloss_h': 'Loss',
              'Vloss_h': 'Loss',
              'sloss_h': 'Loss',
              'alpha_normed_h': 'Normalized Alpha',
              'alpha_normed_biased_h': 'Normalized Alpha minus alpha min',
              'alpha_h': 'Alpha',
              'TlossC_h': 'Loss', 'VlossC_h': 'Loss', 'slossC_h': 'Loss',
              'activeC_h': 'Class #', 'snoozeC_h': 'Class #',
              'Tloss_f': 'Loss', 'Vloss_f': 'Loss',
              'Tnormg_f': 'Norm of gradients', 'Vnormg_f': 'Norm of gradients'}
    ylabel = {'Tacc': 'Training Accuracy (%)', 'Terror': 'Training Error (%)',
              'train/accuracy': 'Training Accuracy (%)',
              'Vacc': 'Test Accuracy (%)', 'Verror': 'Test Error (%)',
              'valid/accuracy': 'Test Accuracy (%)',
              'touch_p': 'Examples Visited (%)',
              'touch': '# Examples Visited',
              'visits_h': '# Examples Visited', 'loss': 'Loss',
              'count_h': '# Examples Waiting', 'epoch': 'Epoch',
              'Tloss': 'Loss', 'Vloss': 'Loss', 'lr': 'Learning rate',
              'train/xent': 'Loss',
              'bumped_p': 'Examples Bumped (%)', 'tau': 'Tau',
              'Tloss_h': '# Examples', 'Vloss_h': '# Examples',
              'alpha_normed_h': '# Examples', 'sloss_h': '# Examples',
              'alpha_normed_biased_h': '# Examples',
              'tauloss_mu': 'Loss', 'tauloss_std': 'Loss std',
              'alpha_h': '# Examples', 'grad_var': 'Gradient Variance',
              'grad_var_n': 'Normalized Gradient Variance',
              'g_variance_mean': 'Mini-batch Gradient Variance',
              'g_norm': 'Step Norm',
              'gbar_norm': 'Gradient Norm',
              'TlossC_h': '# Classes', 'VlossC_h': '# Classes',
              'slossC_h': '# Classes',
              'activeC_h': '# of Actives', 'snoozeC_h': '# of Snoozed',
              'Tloss_f': '# Examples', 'Vloss_f': '# Examples',
              'Tnormg_f': '# Examples', 'Vnormg_f': '# Examples',
              'grad_bias': 'Gradient Diff norm', 'est_var': 'Mean variance',
              'est_snr': 'Mean SNR', 'gb_td': 'Total distortion',
              'gb_cs': 'Cluster size',
              'gb_reinits': '# of Reinits',
              'est_nvar': 'Mean Normalized Variance'}
    titles = {'Tacc': 'Training Accuracy', 'Terror': 'Training Error',
              'train/accuracy': 'Training Accuracy',
              'Vacc': 'Test Accuracy', 'Verror': 'Test Error',
              'valid/accuracy': 'Test Accuracy',
              'touch_p': 'Percentage of Examples Visited per Round',
              'touch': 'Examples Visited per Round',
              'visits_h': 'Histogram of Visits', 'loss': 'Loss',
              'count_h': 'Histogram of Remaining Wait', 'epoch': 'Epoch',
              'Tloss': 'Loss on full training set', 'lr': 'Learning rate',
              'train/xent': 'mini-batch training loss',
              'Vloss': 'Loss on validation set',
              'bumped_p': 'Percentage of Examples Bumped per Round',
              'tau': 'Snooze Threshold',
              'Tloss_h': 'Hostagram of full training set loss',
              'Vloss_h': 'Hostagram of test set loss',
              'sloss_h': 'Hostagram of last loss',
              'alpha_normed_h': 'Histogram of normalized alpha',
              'alpha_normed_biased_h': 'Histogram of normalized biased alpha',
              'tauloss_mu': 'Mean of Loss',
              'tauloss_std': 'Loss standard deviation',
              'alpha_h': 'Histogram of alpha',
              'grad_var': 'Gradient Variance $|\\bar{g}-g|^2/D(g)$',
              'grad_var_n':
              'Normalized Gradient Variance $|\\bar{g}-g|^2/|\\bar{g}|^2$',
              'gbar_norm': 'Norm of Velocity Vector',
              'g_variance_mean':
              'Mini-batch gradient variance $\\sum_B (g-\\bar{g})^2$',
              'g_norm': 'Step Norm',
              'TlossC_h': 'Hostagram of training class loss',
              'VlossC_h': 'Hostagram of test class loss',
              'slossC_h': 'Hostagram of last class loss',
              'activeC_h': 'Histogram of # of actives per class',
              'snoozeC_h': 'Histogram of # of snoozed per class',
              'Tloss_f': 'Histogram of loss (train set, postcomp)',
              'Vloss_f': 'Histogram of loss (val set, postcomp)',
              'Tnormg_f': 'Histogram of $\\|g\\|$ (train set, postcomp)',
              'Vnormg_f': 'Histogram of $\\|g\\|$ (val set, postcomp)',
              'grad_bias': 'Optimization Step Bias',
              'est_var': 'Optimization Step Variance (w/o learning rate)',
              'est_snr': 'Optimization Step SNR',
              'est_nvar': 'Optimization Step Normalized Variance (w/o lr)',
              'gb_td': 'Total distortion of the Gluster objective.',
              'gb_cs': 'Size of Cluster #0',
              'gb_reinits': 'Total number of reinitializations.'}
    yscale_log = ['Tloss', 'Vloss', 'tau', 'lr']  # , 'est_var', 'gb_td'
    yscale_base = ['tau']
    # yscale_sci = ['est_bias', 'est_var', 'gb_td']
    plot_fs = {'Tacc': plot_f, 'Vacc': plot_f,
               'Terror': plot_f, 'Verror': plot_f,
               'Tloss': plot_f, 'Vloss': plot_f,
               'grad_var': plot_smooth_o1, 'grad_var_n': plot_smooth_o1,
               'gbar_norm': plot_smooth_o1, 'g_variance_mean': plot_smooth_o1}
    if 'nolog' in tag_name:
        idx = tag_name.find('_nolog')
        tag_name = tag_name[:idx]+tag_name[idx+6:]
    for name in ['loss', 'acc']:
        if name in tag_name:
            ylabel[tag_name] = ylabel['%s%s' % (tag_name[0], name)]
            titles[tag_name] = titles['%s%s' % (tag_name[0], name)]
    for i in range(10):
        for prefix in ['T', 'V']:
            xlabel['%slossC%d_h' % (prefix, i)] = xlabel['%sloss_h' % prefix]
            ylabel['%slossC%d_h' % (prefix, i)] = ylabel['%sloss_h' % prefix]
            classes = ''
            data0 = data
            if not isinstance(data, list):
                data0 = [data0]
            for d in data0:
                if 'classes' in d and i < len(d['classes']):
                    classes = d['classes'][i]
            titles['%slossC%d_h' % (prefix, i)] = '%s, class %d,%s' % (
                titles['%sloss_h' % prefix], i, classes)
    for k in list(ylabel.keys()):
        if k not in xlabel:
            xlabel[k] = 'Training Iteration'
        xlabel[k + '_Nex'] = '# Examples Processed'
        ylabel[k + '_Nex'] = ylabel[k]
        titles[k + '_Nex'] = titles[k]
        if k not in plot_fs:
            plot_fs[k] = plot_f
        plot_fs[k + '_Nex'] = plot_fs[k]
        xlabel[k + '_log'] = 'Log ' + xlabel[k]
        ylabel[k + '_log'] = ylabel[k]
        titles[k + '_log'] = titles[k]
        if k not in plot_fs:
            plot_fs[k] = plt.plot
        plot_fs[k + '_log'] = plot_fs[k]

    if '_h_' in tag_name and tag_name[-1].isdigit():
        h_index = ' (iter: %d%%)' % int(tag_name[tag_name.rfind('_')+1:])
        tg = tag_name[:tag_name.rfind('_')]
        xlabel[tag_name] = xlabel[tg]
        ylabel[tag_name] = ylabel[tg]
        titles[tag_name] = titles[tg] + h_index
        # if tg in yscale_log:
        #     yscale_log += [tag_name]
    if not isinstance(data, list):
        data = [data]
        run_names = [run_names]

    # color = ['blue', 'orangered', 'limegreen', 'darkkhaki', 'cyan', 'grey']
    import seaborn as sns
    color = sns.color_palette("bright", 6)
    color = color[:ncolor]
    style = ['-', '--', ':', '-.']
    # plt.rcParams.update({'font.size': 12})
    plt.grid(linewidth=1)
    legends = []
    for i in range(len(data)):
        if tag_name not in data[i]:
            continue
        legends += [get_legend(lg_tags, run_names[i], lg_replace)]
        if isinstance(data[i][tag_name], tuple):
            # plt.hist(data[i][tag_name][0], data[i][tag_name][1],
            #          color=color[i])
            edges = data[i][tag_name][1]
            frq = data[i][tag_name][0]
            plt.bar(edges[:-1], frq, width=np.diff(edges),
                    ec="k", align="edge", color=color[color0 + i],
                    alpha=(0.5 if len(data) > 1 else 1))
        else:
            if plot_fs[tag_name].__name__ != plt.plot.__name__:
                plot_fs[tag_name](
                    data[i][tag_name][0], data[i][tag_name][1],
                    linestyle=style[(color0 + i) // len(color)],
                    color=color[(color0 + i) % len(color)], linewidth=2,
                    log_scale=(tag_name in yscale_log),
                    s=smooth_div_n)
            else:
                plot_fs[tag_name](
                    data[i][tag_name][0], data[i][tag_name][1],
                    linestyle=style[(color0 + i) // len(color)],
                    color=color[(color0 + i) % len(color)], linewidth=2)
    if not no_title:
        plt.title(titles[tag_name])
    if tag_name in yscale_log:
        ax = plt.gca()
        if tag_name in yscale_base:
            ax.set_yscale('log', basey=np.e)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
        else:
            ax.set_yscale('log')
    else:
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    if ylim is not None:
        plt.ylim(ylim)
    # plt.xlim([0, 25000])
    plt.legend(legends)
    plt.xlabel(xlabel[tag_name])
    plt.ylabel(ylabel[tag_name])


def ticks(y, pos):
    return r'$e^{{{:.0f}}}$'.format(np.log(y))


def plot_runs_and_tags(get_data_f, plot_f, logdir, patterns, tag_names,
                       fig_name, lg_tags, ylim, batch_size=None, sep_h=True,
                       ncolor=None, save_single=False, lg_replace=[],
                       no_title=False, smooth_div_ns=None):
    if smooth_div_ns is None:
        smooth_div_ns = [None]*len(tag_names)
    run_names = get_run_names(logdir, patterns)
    data = get_data_f(logdir, run_names, tag_names, batch_size)
    if len(data) == 0:
        return data, run_names
    # plt.figure(figsize=(26,14))
    # nruns = len(run_names)
    # nruns = 2
    # nruns = len(data)
    num = 0
    # dm = []
    for tg in tag_names:
        if ('_h' in tg or '_f' in tg) and sep_h:
            for i in range(len(data)):
                if tg in data[i]:
                    num += 1
        else:
            num += 1
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    if save_single:
        plt.rcParams.update({'font.size': 16})
    else:
        plt.rcParams.update({'font.size': 12})
    if not save_single:
        fig = plt.figure(figsize=(7 * width, 4 * height))
        fig.subplots(height, width)
    else:
        plt.figure(figsize=(7, 4))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    if save_single:
        fig_dir = fig_name[:fig_name.rfind('.')]
        try:
            os.makedirs(fig_dir)
        except os.error:
            pass
    for i in range(len(tag_names)):
        yl = ylim[i]
        if ('_h' in tag_names[i] or '_f' in tag_names[i]) and sep_h:
            for j in range(len(data)):
                if tag_names[i] not in data[j]:
                    continue
                if not save_single:
                    plt.subplot(height, width, fi)
                plot_tag(data[j], plot_f, run_names[j],
                         tag_names[i], lg_tags, yl, color0=j, ncolor=ncolor,
                         lg_replace=lg_replace, no_title=no_title,
                         smooth_div_n=smooth_div_ns[i])
                if save_single:
                    # plt.savefig('%s/%s.png' % (fig_dir, tag_names[i]),
                    plt.savefig('%s/%s.pdf' % (fig_dir, tag_names[i]),
                                dpi=100, bbox_inches='tight')
                    ax = plt.gca()
                    ax.get_legend().remove()
                    plt.savefig('%s/%s_nolegend.pdf' % (fig_dir, tag_names[i]),
                                dpi=100, bbox_inches='tight')
                    plt.figure(figsize=(7, 4))
                fi += 1
        else:
            if not isinstance(yl, list) and yl is not None:
                yl = ylim
            if not save_single:
                plt.subplot(height, width, fi)
            plot_tag(data, plot_f, run_names, tag_names[i], lg_tags, yl,
                     ncolor=ncolor, lg_replace=lg_replace, no_title=no_title,
                     smooth_div_n=smooth_div_ns[i])
            if save_single:
                # plt.savefig('%s/%s.png' % (fig_dir, tag_names[i]),
                plt.savefig('%s/%s.pdf' % (fig_dir, tag_names[i]),
                            dpi=100, bbox_inches='tight')
                ax = plt.gca()
                ax.get_legend().remove()
                plt.savefig('%s/%s_nolegend.pdf' % (fig_dir, tag_names[i]),
                            dpi=100, bbox_inches='tight')
                plt.figure(figsize=(7, 4))
            fi += 1
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names


def plot_clusters(run_dir, gfname, nsamples=20, sz=28, seed=123, topk=True):
    gpath = os.path.join(run_dir, '%s.pth.tar' % gfname)
    fig_name = os.path.join(run_dir, '%s.png' % gfname)
    data = torch.load(gpath)
    assign_i, target_i = data['assign'], data['target']
    pred_i, loss_i, normC = data['pred'], data['loss'], data['normC']
    total_dist = data['total_dist']
    topk_i = data['topk']
    if topk:
        correct = topk_i[:, 1]
    else:
        correct = topk_i[:, 0]
    opt, dataset = data['opt'], data['dataset']
    opt = utils.DictWrapper(opt)
    pad = 2
    np.random.seed(seed)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    train_loader, test_loader, train_test_loader = get_loaders(opt)
    if dataset == 'train_test':
        dataset = train_test_loader.dataset
    elif dataset == 'test':
        dataset = test_loader.dataset
    dataset.ds.transform = transforms.Compose([
        transforms.Resize(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        ])

    _, counts = np.unique(assign_i, return_counts=True)

    nclusters = int(assign_i.max()+1)
    height = nclusters
    width = nsamples
    plt.figure(figsize=(7 * width, 4 * height))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)

    lc = np.zeros((nclusters, ))
    for i in range(nclusters):
        idx, _ = np.where(assign_i == i)
        if len(idx) > 0:
            lc[i] = loss_i[idx].mean()
    cids = np.argsort(lc)
    hpad = sz+2*pad
    images = np.zeros((nclusters, nsamples, 3, hpad, hpad))
    print('Total loss: %.4f' % (loss_i.mean()))
    print('Total accuracy: %.2f%%' % ((correct*100.).mean()))
    print('Total dist: %.4f' % total_dist.sum())
    print('')
    for c in range(nclusters):
        i = cids[c]
        idx, _ = np.where(assign_i == i)
        print('Cluster %d, size: %d' % (i, len(idx)))
        if len(idx) == 0:
            continue
        print('Cluster %d, normC: %.8f' % (i, normC[i]))
        print('Cluster %d, loss: %.4f' % (i, loss_i[idx].mean()))
        acc = (correct[idx]*100.).mean()
        print('Cluster %d, accuracy: %.2f%%\n' % (i, acc))
        np.random.shuffle(idx)
        for j in range(min(len(idx), nsamples)):
            xi = dataset[idx[j]][0]
            xi2 = np.zeros((3, hpad, hpad))
            xi2[0] = (pred_i[idx[j]] != target_i[idx[j]])
            if xi.shape[0] == 1:
                xi2[0:1, pad:sz+pad, pad:sz+pad] = xi
                xi2[1:2, pad:sz+pad, pad:sz+pad] = xi
                xi2[2:3, pad:sz+pad, pad:sz+pad] = xi
            else:
                xi2[:, pad:sz+pad, pad:sz+pad] = xi
            images[c, j] = xi2

    fig = torch.tensor(images.reshape(-1, 3, hpad, hpad))
    xi = vutils.make_grid(fig, nrow=nsamples, normalize=True, scale_each=True)

    plt.imshow(xi.numpy().transpose([1, 2, 0]))
    plt.axis('off')
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')


def plot_one_iter(
        nclusters, nsamples, dataset, sz, pad, hpad,
        assign_i, target_i, pred_i, loss_i, delim=True):
    # nclusters = int(assign_i.max()+1)
    lc = np.zeros((nclusters, ))
    for i in range(nclusters):
        idx = np.where(assign_i == i)[0]
        if len(idx) > 0:
            lc[i] = loss_i[idx].mean()
    # cids = np.argsort(lc)
    images = np.zeros((nclusters, nsamples, 3, hpad, hpad))
    for c in range(nclusters):
        # i = cids[c]
        i = c
        idx = np.where(assign_i == i)[0]
        if len(idx) == 0:
            continue
        np.random.shuffle(idx)
        for j in range(min(len(idx), nsamples)):
            xi = dataset[idx[j]][0]
            xi2 = np.zeros((3, hpad, hpad))
            xi2[0] = (pred_i[idx[j]] != target_i[idx[j]])
            if xi.shape[0] == 1:
                xi2[0:1, pad:sz+pad, pad:sz+pad] = xi
                xi2[1:2, pad:sz+pad, pad:sz+pad] = xi
                xi2[2:3, pad:sz+pad, pad:sz+pad] = xi
            else:
                xi2[:, pad:sz+pad, pad:sz+pad] = xi
            images[c, j] = xi2
    # if delim:
    #     images[:, -1, :, :, -pad:] = 1
    return images


def plot_clusters_online(
        run_dir, gfname, nsamples=20, iters=[1], sz=28, seed=123, topk=True):
    gpath = os.path.join(run_dir, '%s.pth.tar' % gfname)
    fig_name = os.path.join(run_dir, '%s.png' % gfname)
    data = torch.load(
            gpath,
            map_location=lambda storage, loc: storage)
    opt = data['opt']
    # dataset = opt['dataset']
    data = data['gvar']['snap_data']
    assign_i, target_i = data['assign'], data['target']
    pred_i, loss_i = data['pred'], data['loss']
    niters = data['niters']
    # total_dist = data['total_dist']
    # correct = pred_i == target_i
    opt = utils.DictWrapper(opt)
    np.random.seed(seed)
    print(niters)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    train_loader, test_loader, train_test_loader = get_loaders(opt)
    dataset = train_test_loader.dataset
    dataset.ds.transform = transforms.Compose([
        transforms.Resize(sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        ])

    nclusters = int(max([a.max()+1 for a in assign_i]))
    height = nclusters
    width = nsamples*len(iters)
    plt.figure(figsize=(7 * width, 4 * height))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)

    pad = 2
    hpad = sz+2*pad
    images = []
    for i in range(len(iters)):
        si = iters[i]
        images += [plot_one_iter(
                nclusters, nsamples, dataset, sz, pad, hpad,
                assign_i[si], target_i, pred_i[si], loss_i[si],
                delim=(i != len(iters)-1))]
    images = np.concatenate(images, axis=1).transpose([1, 0, 2, 3, 4])

    fig = torch.tensor(images.reshape(-1, 3, hpad, hpad))
    xi = vutils.make_grid(fig, nrow=height, normalize=True, scale_each=True)

    xi = xi.numpy().transpose([1, 2, 0])
    xi2 = np.zeros((xi.shape[0]+(len(iters)-1)*pad, xi.shape[1], xi.shape[2]))
    for i in range(len(iters)):
        ww = xi.shape[0]/len(iters)
        xi2[(ww+pad)*i:(ww+pad)*i+ww, :, :] = xi[ww*i:ww*(i+1), :, :]
        if i != len(iters)-1:
            xi2[(ww+pad)*i+ww:(ww+pad)*(i+1), :, :] = 1
    plt.imshow(xi2)
    plt.axis('off')
    plt.savefig(fig_name, dpi=20, bbox_inches='tight')


def aggregate_data(data, run_names, plot_tg, x_tg, y_tg, lg_tags):
    agg_run_names = OrderedDict()
    agg_data = OrderedDict()
    x_tags = x_tg.split('/')
    # aggregate data along any tag not in x_tg or lg_tags
    keep_tags = x_tags + lg_tags
    for i in range(len(data)):
        if plot_tg not in data[i]:
            continue
        run_name = run_names[i]
        agr = get_legend(keep_tags, run_name)
        if agr not in agg_run_names:
            agg_run_names[agr] = []
            agg_data[agr] = []
        agg_run_names[agr] += [run_name]
        agg_data[agr] += [data[i][plot_tg][-1]]

    # Make a curve out of aggregated points
    agg_data_curve = OrderedDict()
    for key, val in agg_data.items():
        def get_xtag_val(x_tg_i, key):
            x_tag_keyval = key[key.find(x_tg_i):]
            if x_tag_keyval.find(',') > -1:
                x_tag_keyval = x_tag_keyval[:x_tag_keyval.find(',')]
            x_tag_key = x_tag_keyval[:x_tag_keyval.rfind('_')]
            x_tag_val = x_tag_keyval[x_tag_keyval.rfind('_')+1:]
            mg = re.search(x_tag_keyval, key)
            key = key.replace(mg.group(0), x_tag_key)
            return x_tag_val, key
        if len(x_tags) == 1:
            x_tag_val, key2 = get_xtag_val(x_tg, key)
        else:
            x_tag_val1, key2 = get_xtag_val(x_tags[0], key)
            x_tag_val2, key2 = get_xtag_val(x_tags[1], key2)
            x_tag_val = float(x_tag_val1)/float(x_tag_val2)
        if key2 not in agg_data_curve:
            agg_data_curve[key2] = []
        agg_data_curve[key2] += [(float(x_tag_val), np.mean(val), np.std(val))]

    for key, val in agg_data_curve.items():
        val.sort()
    agg_run_names_curve = list(agg_data_curve.keys())
    return agg_data_curve, agg_run_names_curve


def plot_tag_multi(data, plot_f, run_names,
                   plot_tg, x_tg, y_tg, lg_tags, ylim=None,
                   ncolor=None, lg_replace=[], no_title=False, color0=0,
                   agg_type='mean'):
    xlabel = {'zeta': '$\\zeta=d/n$',
              'psi': '$\\psi=p/n$',
              'psi1': '$\\psi_1=h/d$',
              'psi2': '$\\psi_2=n/d$',
              'student_hidden': '$h_s$',
              'teacher_hidden': '$h_t$',
              'num_train_data': 'N'}
    ylabel = {'est_var': 'Variance', 'est_nvar': 'Normalized Variance'}
    titles = {'risk_logistic': 'Logistic Risk',
              'risk_zero_one': '0/1 Risk',
              'risk_l2': '$L_2$ Risk',
              'est_var': 'Optimization Step Variance (w/o learning rate)',
              'est_nvar': 'Optimization Step Normalized Variance (w/o lr)'}
    yscale_log = ['est_var']  # 'est_nvar']
    plot_fs = {}
    if plot_tg not in plot_fs:
        plot_fs[plot_tg] = plot_f

    if not isinstance(data, list):
        data = [data]
        run_names = [run_names]
    for d in data:
        for tag_name in d.keys():
            if agg_type == 'min':
                min_idx = np.argmin(d[tag_name][1])
                d[tag_name] = d[tag_name][:, min_idx]
            elif agg_type == 'max':
                max_idx = np.argmax(d[tag_name][1])
                d[tag_name] = d[tag_name][:, max_idx]
            elif agg_type == 'end':
                d[tag_name] = d[tag_name][:, -1]
            elif agg_type == 'mean':
                d[tag_name] = d[tag_name].mean(1)
            elif 'max_' in agg_type:
                import math
                start = int(math.ceil(d[tag_name].shape[1]
                                      * (1 - float(agg_type[4:])/100)))
                max_idx = np.argmax(d[tag_name][1][start:])+start
                d[tag_name] = d[tag_name][:, max_idx]
            elif 'mean_' in agg_type:
                import math
                start = int(math.ceil(d[tag_name].shape[1]
                                      * (1 - float(agg_type[5:])/100)))
                d[tag_name] = d[tag_name][:, start:].mean(1)
    agg_data, agg_run_names = aggregate_data(
        data, run_names, plot_tg, x_tg, y_tg, lg_tags)

    # color = ['blue', 'orangered', 'limegreen', 'darkkhaki', 'cyan', 'grey']
    import seaborn as sns
    color = sns.color_palette("bright", 6)
    color = color[:ncolor]
    style = ['-', '--', ':', '-.']
    plt.grid(linewidth=1)
    legends = []
    for i, (key, val) in enumerate(agg_data.items()):
        legends += [get_legend(lg_tags, agg_run_names[i], lg_replace)]
        val = np.array(val)
        plot_fs[plot_tg](
            val[:, 0], val[:, 1],
            linestyle=style[(color0 + i) // len(color)],
            color=color[(color0 + i) % len(color)], linewidth=2)
        plt.errorbar(val[:, 0], val[:, 1], yerr=val[:, 2],
                     marker='o', linestyle='', capsize=5,
                     color=color[(color0 + i) % len(color)])
    if not no_title:
        agg0 = agg_type.replace('_', ' ')
        agg0 = agg0[0].upper() + agg0[1:]
        plt.title('%s (%s%%)' % (titles[plot_tg], agg0))
    if plot_tg in yscale_log:
        ax = plt.gca()
        ax.set_yscale('log')
    else:
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax = plt.gca()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(legends)
    xlabelstr = ''
    for x_tg_i in x_tg.split('/'):
        if len(xlabelstr) > 0:
            xlabelstr += '/'
        xlabelstr += xlabel[x_tg_i]
    plt.xlabel(xlabelstr)
    agg0 = agg_type.split('_')[0]
    agg0 = agg0[0].upper() + agg0[1:]
    plt.ylabel(agg0 + ' ' + ylabel[y_tg])


def plot_runs_and_tags_multi(plot_f, logdir, patterns,
                             plot_tags, x_tags, y_tags, lg_tags,
                             fig_name, ylim,
                             ncolor=None, lg_replace=[], save_single=False,
                             no_title=False, agg_types=[]):
    y_tags = plot_tags
    run_names = get_run_names(logdir, patterns)
    data = get_data_pth(logdir, run_names, plot_tags)
    if len(data) == 0:
        return data, run_names
    num = len(plot_tags)
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    if not save_single:
        fig = plt.figure(figsize=(7 * width, 4 * height))
        fig.subplots(height, width)
    else:
        plt.figure(figsize=(7, 4))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    if save_single:
        fig_dir = fig_name[:fig_name.rfind('.')]
        try:
            os.makedirs(fig_dir)
        except os.error:
            pass
    for i in range(len(plot_tags)):
        yl = ylim[i]
        if not isinstance(yl, list) and yl is not None:
            yl = ylim
        if not save_single:
            plt.subplot(height, width, fi)
        plot_tag_multi(copy.deepcopy(data), plot_f, run_names,
                       plot_tags[i], x_tags[i], y_tags[i], lg_tags[i], yl,
                       ncolor=ncolor, lg_replace=lg_replace, no_title=no_title,
                       agg_type=agg_types[i])
        if save_single:
            # plt.savefig('%s/%s.png' % (fig_dir, tag_names[i]),
            plt.savefig('%s/%s_%s.pdf' % (fig_dir, plot_tags[i], agg_types[i]),
                        dpi=100, bbox_inches='tight')
            ax = plt.gca()
            ax.get_legend().remove()
            plt.savefig('%s/%s_%s_nolegend.pdf' % (fig_dir, plot_tags[i],
                                                   agg_types[i]),
                        dpi=100, bbox_inches='tight')
            plt.figure(figsize=(7, 4))
        fi += 1
    if not save_single:
        plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names
