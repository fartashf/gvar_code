# using event multiplexer from tensorboard
import tensorboard_extract as tb_extract
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # NOQA
from scipy.interpolate import spline
import numpy as np
import os
import re
import requests
import torch
import pylab as plt
import matplotlib.ticker as mtick


class TBMultiplexer:

    def __init__(self, logdir):
        self.logdir = logdir
        SIZE_GUIDANCE = {'scalars': 1000}
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
        logdata = torch.load(run_name + '/log.pth.tar')
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
            if tag_name not in logdata:
                continue
            js = logdata[tag_name]
            if '_h' in tag_name:
                d[tag_name] = js[-1][2]
                if tag_name_orig_log is not None:
                    d[tag_name_orig_log] = (d[tag_name][0],
                                            np.exp(d[tag_name][1]))
            else:
                d[tag_name] = np.array([[x[j] for x in js]
                                        for j in range(1, 3)])
                if tag_name_orig_nex is not None:
                    d[tag_name_orig_nex] = np.array(d[tag_name])
                    d[tag_name_orig_nex][0] = d[tag_name][0] * batch_size
        data += [d]
    return data


def plot_smooth(x, y, npts=100, order=3, *args, **kwargs):
    x_smooth = np.linspace(x.min(), x.max(), npts)
    y_smooth = spline(x, y, x_smooth, order=order)
    # x_smooth = x
    # y_smooth = y
    plt.plot(x_smooth, y_smooth, *args, **kwargs)


def plot_smooth_o1(x, y, *args, **kwargs):
    plot_smooth(x, y, 100, 1, *args, **kwargs)


def plot_tag(data, plot_f, run_names, tag_name, lg_tags, ylim=None, color0=0,
             ncolor=None):
    xlabel = {'visits_h': '# Visits',
              'count_h': '# Wait Iterations',
              'Tloss_h': 'Loss',
              'Vloss_h': 'Loss',
              'sloss_h': 'Loss',
              'alpha_normed_h': 'Normalized Alpha',
              'alpha_h': 'Alpha'}
    ylabel = {'Tacc': 'Training Accuracy (%)',
              'Vacc': 'Test Accuracy (%)',
              'touch_p': 'Examples Visited (%)',
              'touch': '# Examples Visited',
              'visits_h': '# Examples Visited', 'loss': 'Loss',
              'count_h': '# Examples Waiting', 'epoch': 'Epoch',
              'Tloss': 'Loss', 'Vloss': 'Loss', 'lr': 'Learning rate',
              'bumped_p': 'Examples Bumped (%)', 'tau': 'Tau',
              'Tloss_h': '# Examples', 'Vloss_h': '# Examples',
              'alpha_normed_h': '# Examples', 'sloss_h': '# Examples',
              'tauloss_mu': 'Loss', 'tauloss_std': 'Loss std',
              'alpha_h': '# Examples', 'grad_var': 'Gradient Variance',
              'grad_var_n': 'Normalized Gradient Variance',
              'gbar_norm': 'Gradient Norm'}
    titles = {'Tacc': 'Training Accuracy',
              'Vacc': 'Test Accuracy',
              'touch_p': 'Percentage of Examples Visited per Round',
              'touch': 'Examples Visited per Round',
              'visits_h': 'Histogram of Visits', 'loss': 'Loss',
              'count_h': 'Histogram of Remaining Wait', 'epoch': 'Epoch',
              'Tloss': 'Loss on full training set', 'lr': 'Learning rate',
              'Vloss': 'Loss on validation set',
              'bumped_p': 'Percentage of Examples Bumped per Round',
              'tau': 'Snooze Threshold',
              'Tloss_h': 'Hostagram of full training set loss',
              'Vloss_h': 'Hostagram of test set loss',
              'sloss_h': 'Hostagram of last loss',
              'alpha_normed_h': 'Histogram of normalized alpha',
              'tauloss_mu': 'Mean of Loss',
              'tauloss_std': 'Loss standard deviation',
              'alpha_h': 'Histogram of alpha',
              'grad_var': 'Gradient Variance $|\\bar{g}-g|^2/D(g)$',
              'grad_var_n':
              'Normalized Gradient Variance $|\\bar{g}-g|^2/|\\bar{g}|^2$',
              'gbar_norm': 'Norm of Velocity Vector'}
    yscale_log = ['Tloss', 'Vloss', 'tau']
    yscale_base = ['tau']
    plot_fs = {'Tacc': plot_f, 'Vacc': plot_f,
               'Tloss': plot_f, 'Vloss': plot_f,
               'grad_var': plot_smooth_o1, 'grad_var_n': plot_smooth_o1,
               'gbar_norm': plot_smooth_o1}
    if 'nolog' in tag_name:
        idx = tag_name.find('_nolog')
        tag_name = tag_name[:idx]+tag_name[idx+6:]
    for k in list(ylabel.keys()):
        if k not in xlabel:
            xlabel[k] = 'Training Iteration'
        xlabel[k + '_Nex'] = '# Examples Processed'
        ylabel[k + '_Nex'] = ylabel[k]
        titles[k + '_Nex'] = titles[k]
        if k not in plot_fs:
            plot_fs[k] = plt.plot
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

    color = ['blue', 'orangered', 'limegreen', 'darkkhaki', 'cyan', 'grey']
    color = color[:ncolor]
    style = ['-', '--', ':', '-.']
    plt.rcParams.update({'font.size': 12})
    plt.grid(linewidth=1)
    legends = []
    for i in range(len(data)):
        if tag_name not in data[i]:
            continue
        lg = ""
        for lgt in lg_tags:
            res = ".*?($|,)" if ',' not in lgt and '$' not in lgt else ''
            mg = re.search(lgt + res, run_names[i])
            if mg:
                lg += mg.group(0)
        legends += [lg]
        if isinstance(data[i][tag_name], tuple):
            # plt.hist(data[i][tag_name][0], data[i][tag_name][1],
            #          color=color[i])
            edges = data[i][tag_name][1]
            frq = data[i][tag_name][0]
            plt.bar(edges[:-1], frq, width=np.diff(edges),
                    ec="k", align="edge", color=color[color0 + i],
                    alpha=(0.5 if len(data) > 1 else 1))
        else:
            plot_fs[tag_name](
                data[i][tag_name][0], data[i][tag_name][1],
                style[(color0 + i) // len(color)],
                color=color[(color0 + i) % len(color)], linewidth=2)
    plt.title(titles[tag_name])
    if tag_name in yscale_log:
        ax = plt.gca()
        if tag_name in yscale_base:
            ax.set_yscale('log', basey=np.e)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
        else:
            ax.set_yscale('log')
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
                       ncolor=None):
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
        if '_h' in tg and sep_h:
            for i in range(len(data)):
                if tg in data[i]:
                    num += 1
        else:
            num += 1
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    fig = plt.figure(figsize=(7 * width, 4 * height))
    fig.subplots(height, width)
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    for i in range(len(tag_names)):
        yl = ylim[i]
        if '_h' in tag_names[i] and sep_h:
            for j in range(len(data)):
                if tag_names[i] not in data[j]:
                    continue
                plt.subplot(height, width, fi)
                plot_tag(data[j], plot_f, run_names[j],
                         tag_names[i], lg_tags, yl, color0=j, ncolor=ncolor)
                fi += 1
        else:
            if not isinstance(yl, list) and yl is not None:
                yl = ylim
            plt.subplot(height, width, fi)
            plot_tag(data, plot_f, run_names, tag_names[i], lg_tags, yl,
                     ncolor=ncolor)
            fi += 1
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names
