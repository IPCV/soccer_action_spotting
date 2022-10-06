import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from pylab import annotate
from addict import Dict

SoccerNetV2 = Dict({'classes': ["Ball out",
                                "Throw-in",
                                "Foul",
                                "Ind. free-kick",
                                "Clearance",
                                "Shots on tar.",
                                "Shots off tar.",
                                "Corner",
                                "Substitution",
                                "Kick-off",
                                "Yellow card",
                                "Offside",
                                "Dir. free-kick",
                                "Goal",
                                "Penalty",
                                "Yel.->Red",
                                "Red card"],
                    'mapping': [8, 9, 10, 11, 7, 5, 6, 13, 3, 1, 14, 4, 12, 2, 0, 16, 15],
                    'num_classes': 17})


def plot_datasets_summary(stats, figsize=None, ylabel="Number of Instances", xlabel="Categories", annot_rotation=90,
                          annot_fontsize=9, axis_fontsize=12, legend=False, width=0.5, annotate_cols=True, title=None,
                          colormap=None):
    if not figsize:
        figsize = (1, 1)

    sns.set_style("whitegrid", {'grid.color': '.9',
                                'axes.edgecolor': '.2',
                                'axes.linewidth': 1})

    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

    if title:
        plt.title(title)

    if stats.shape[1] == 1 and not colormap:
        stats.plot(kind='bar', ax=ax, legend=legend, width=width, color=[sns.color_palette("PuBu", 10)[-3]])
    else:
        stats.plot(kind='bar', ax=ax, legend=legend, width=width, colormap=colormap)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=annot_rotation, fontsize=axis_fontsize, ha='right',
             rotation_mode="anchor")
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=axis_fontsize)

    if annotate_cols:
        for i, (index, row) in enumerate(stats.iterrows()):
            for col in stats.columns.values:
                annotate('{:.0f}'.format(row[col]), (i, row[col] + 25), ha='center', fontsize=annot_fontsize)
        plt.ylim([0, 1.1 * stats.values.max()])

    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    return fig, ax


def plot_soccernet_histogram(dataset, map_categories=True, figsize=None):
    if map_categories:
        categories = SoccerNetV2.classes
        mapping = SoccerNetV2.mapping
        histogram = pd.DataFrame(data=dataset.histograms[mapping, -1], index=categories)
    else:
        categories = [c for c in dataset.classes.values() if c != 'Background']
        mapping = np.arange(len(dataset.classes) - 1)
        histogram = pd.DataFrame(data=dataset.histograms[mapping, -1], index=categories)

    sns.color_palette()

    if not figsize:
        figsize = (16, 4)

    my_cmap = ListedColormap(sns.color_palette("hls", 1).as_hex())
    fig, ax = plot_datasets_summary(histogram, figsize=figsize,
                                    legend=False,
                                    width=0.75,
                                    annotate_cols=False,
                                    annot_rotation=35,
                                    colormap=my_cmap,
                                    title='SoccerNet V2')

    for i, (index, row) in enumerate(histogram.iterrows()):
        for j, col in enumerate(histogram.columns.values):
            annotate('{:,.0f}'.format(dataset.histograms[mapping[i], -1]),
                     (i + 0.225 * -0.07, 200 + row[col]),
                     ha='center', va='bottom', fontsize=12, rotation=0)

    plt.ylim([0, 1.3 * histogram.values.max()])
    plt.yscale('symlog', linthresh=20000)

    yticks = np.concatenate((np.arange(1000, 7000, step=2000),
                             np.arange(10000, 19000, step=5000),
                             np.arange(20000, 31000, step=5000)))
    ylabels = ["{:,}".format(v) for v in yticks]
    plt.yticks(yticks, ylabels)
    return fig, ax


def _plot_differential(fig, ax_plot, flow, title=None, scale=False, center=False, ax_cbar=None, cbar_legends=None):
    if title:
        ax_plot.set_title(title, fontweight="bold")

    if not scale:
        min_, max_ = flow.min(), flow.max()
    else:
        min_, max_ = 0, 1

    if center:
        max_abs = np.max((np.abs(min_), np.abs(max_)))
        min_, max_ = -max_abs, max_abs

    amap = ax_plot.imshow(flow, interpolation='nearest', vmin=min_, vmax=max_)

    if not ax_cbar:
        return

    if cbar_legends:
        ticks = np.linspace(min_, max_, 7)
        cbar = fig.colorbar(amap, cax=ax_cbar, ticks=ticks)
        ticks = ['{:.2}'.format(t) for t in ticks]
        ticks[0], ticks[-1] = cbar_legends
        cbar.ax.set_yticklabels(ticks)
    else:
        fig.colorbar(amap, cax=ax_cbar)


def plot_flow(prev_frame, next_frame, flow, scale=False, center=False, figsize=(10, 10), title=None):
    gs = gridspec.GridSpec(2, 6, width_ratios=[5, 0.5, 0.85, 5, 0.5, 0.85])
    fig = plt.figure(figsize=figsize)

    if title:
        fig.subplots_adjust(top=0.7)
        fig.suptitle(title, fontweight="bold", fontsize=14, y=0.75)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Previous Frame", fontweight="bold")
    ax1.imshow(prev_frame)

    ax2 = fig.add_subplot(gs[3])
    ax2.set_title("Next Frame", fontweight="bold")
    ax2.imshow(next_frame)

    _plot_differential(fig, fig.add_subplot(gs[6]), flow[:, :, 0], 'dU', scale, center, fig.add_subplot(gs[7]),
                       ('LEFT', 'RIGHT'))

    _plot_differential(fig, fig.add_subplot(gs[9]), flow[:, :, 1], 'dV', scale, center, fig.add_subplot(gs[10]),
                       ('UP', 'DOWN'))
    return fig
