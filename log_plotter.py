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
            if 'Nex' in tag_name and tag_name not in logdata:
                tag_name_orig = tag_name
                tag_name = tag_name[:-4]
            else:
                tag_name_orig = None
            if tag_name not in logdata:
                continue
            js = logdata[tag_name]
            if '_h' in tag_name:
                d[tag_name] = js[-1][2]
            else:
                d[tag_name] = np.array([[x[j] for x in js]
                                        for j in range(1, 3)])
                if tag_name_orig is not None:
                    d[tag_name_orig] = np.array(d[tag_name])
                    d[tag_name_orig][0] = d[tag_name][0] * batch_size
        data += [d]
    return data


def plot_smooth(x, y, *args, **kwargs):
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = spline(x, y, x_smooth)
    # x_smooth = x
    # y_smooth = y
    plt.plot(x_smooth, y_smooth, *args, **kwargs)


def plot_tag(data, plot_f, run_names, tag_name, lg_tags, ylim=None, color0=0):
    xlabel = {'visits_h': '# Visits',
              'count_h': '# Wait Iterations'}
    ylabel = {'Tacc': 'Training Accuracy (%)',
              'Vacc': 'Test Accuracy (%)',
              'touch_p': 'Examples Visited (%)',
              'visits_h': '# Examples Visited', 'loss': 'Loss',
              'count_h': '# Examples Waiting', 'epoch': 'Epoch',
              'Tloss': 'Loss', 'lr': 'Learning rate',
              'bumped_p': 'Examples Bumped (%)', 'tau': 'Tau'}
    titles = {'Tacc': 'Training Accuracy',
              'Vacc': 'Test Accuracy',
              'touch_p': 'Percentage of Examples Visited per Round',
              'visits_h': 'Histogram of Visits', 'loss': 'Loss',
              'count_h': 'Histogram of Remaining Wait', 'epoch': 'Epoch',
              'Tloss': 'Loss on full training set', 'lr': 'Learning rate',
              'bumped_p': 'Percentage of Examples Bumped per Round',
              'tau': 'Snooze Threshold'}
    plot_fs = {'Tacc': plot_f, 'Vacc': plot_f, 'Tloss': plot_f}
    for k in list(ylabel.keys()):
        if k not in xlabel:
            xlabel[k] = 'Training Iteration'
        xlabel[k + '_Nex'] = '# Examples Processed'
        ylabel[k + '_Nex'] = ylabel[k]
        titles[k + '_Nex'] = titles[k]
        if k not in plot_fs:
            plot_fs[k] = plt.plot
        plot_fs[k + '_Nex'] = plot_fs[k]

    if not isinstance(data, list):
        data = [data]
        run_names = [run_names]

    color = ['blue', 'orangered', 'limegreen', 'darkkhaki', 'cyan', 'black']
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
                    ec="k", align="edge", color=color[color0 + i])
        else:
            plot_fs[tag_name](
                data[i][tag_name][0], data[i][tag_name][1],
                style[(color0 + i) // len(color)],
                color=color[(color0 + i) % len(color)], linewidth=2)
    plt.title(titles[tag_name])
    if ylim is not None:
        plt.ylim(ylim)
    # plt.xlim([0, 25000])
    plt.legend(legends)
    plt.xlabel(xlabel[tag_name])
    plt.ylabel(ylabel[tag_name])


def plot_runs_and_tags(get_data_f, plot_f, logdir, patterns, tag_names,
                       fig_name, lg_tags, ylim, batch_size=None):
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
        if tg[-2:] != '_h':
            num += 1
        else:
            for i in range(len(data)):
                if tg in data[i]:
                    num += 1
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    fig = plt.figure(figsize=(7 * width, 4 * height))
    fig.subplots(height, width)
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    for i in range(len(tag_names)):
        if tag_names[i][-2:] == '_h':
            for j in range(len(data)):
                if tag_names[i] not in data[j]:
                    continue
                plt.subplot(height, width, fi)
                plot_tag(data[j], plot_f, run_names[j],
                         tag_names[i], lg_tags, color0=j)
                fi += 1
        else:
            yl = ylim[i]
            if not isinstance(yl, list) and yl is not None:
                yl = ylim
            plt.subplot(height, width, fi)
            plot_tag(data, plot_f, run_names, tag_names[i], lg_tags, yl)
            fi += 1
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names
