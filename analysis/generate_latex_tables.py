import argparse
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import os
import pandas as pd
# from datareader import aggregate, read_mace_distribution, POS_LABEL_MAP
# from scipy.special import softmax
# from scipy.spatial.distance import jensenshannon
# from scipy.special import kl_div
# from util.temperature_scaling import JSD
# import torch
# import torch.nn.functional as F
# from scipy.stats import pearsonr
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import kruskal
import ipdb


font = {'size'   : 18}

matplotlib.rc('font', **font)

METRICS_TABLE_HEAD = """
\\begin{table*}%[htp]
    %\\setlength{\\tabcolsep}{1.5pt}
    \\def\\arraystretch{1.2}
    \\centering
    \\fontsize{10}{10}\\selectfont
    \\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c | c c | c c | c c}
    \\toprule %\\thickhline
    & \\multicolumn{2}{c}{ {d0} } & \\multicolumn{2}{c}{ {d1} } & \\multicolumn{2}{c}{ {d2} } & \\multicolumn{2}{c}{ {d3} }\\\\
    \\midrule
    Method & F1 & CLL & F1 & CLL & F1 & CLL & F1 & CLL\\\\
"""

METRICS_TABLE_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{F1 and calibrated log likelihood. Results are averaged over 10 random seeds; standard deviation is given in the subscript.} %Results are reported as the average macro F1 over 5 random seeds.}
    \\label{tab:{dataset}_results}
\\end{table*}
"""


CORRELATION_TABLE_HEAD = """
\\begin{table}%[htp]
    %\\setlength{\\tabcolsep}{1.5pt}
    \\def\\arraystretch{1.2}
    \\centering
    \\fontsize{10}{10}\\selectfont
    \\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c c c}
    \\toprule %\\thickhline
    Dataset & Avg & Centroid & Temp & Hybrid \\\\
    \\midrule 
"""

CORRELATION_TABLE_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Pearon correlation between the JS divergence of different aggregation methods and individual methods correlated with the raw performance of the individual methods ($r_{p}$) and the individual method's average JS divergence to each other individual method ($r_{\mu}$). **indicates significance at p < 0.05.} %Results are reported as the average macro F1 over 5 random seeds.}
    \\label{tab:correlation_results}
\\end{table}
"""

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def change_height(ax, new_value) :
    for patch in ax.patches :
        current_height = patch.get_height()
        diff = current_height - new_value

        # we change the bar width
        patch.set_height(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)


def plot_metrics_bayesian(plot_dframes, metric_name, method_map, best_per_group, group_name_map, mu, outfile, dataset, add_y_labels=True):
    bar_chart_dframe = pd.DataFrame(plot_dframes[metric_name], columns=[metric_name, 'split', 'Experiment'])
    alpha = 0.65#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    lab_colors = [['black' for i in range(len(method_map[m]))] for m in method_map]
    font_style = [[[] for i in range(len(method_map[m]))] for m in method_map]
    best_color = 'black'
    #orig_palette = sns.color_palette()
    #palette = [orig_palette[0]]*4 + [orig_palette[1]] * 3 + [orig_palette[2]] * 2
    palette = [[method[2] for method in method_map[m]] for m in method_map]
    round_digits = 3
    if metric_name in ['P', 'R', 'F1']:
        # plt_min = 30
        # plt_max = 80
        xlim = dataset[2]
        #xloc = 33.5
    else:
        # plt_min = 0.3
        # plt_max = 0.9
        xlim = dataset[3]
        #xloc = 0.335

    # Get the label xloc
    all_vals = []
    for g,group in bar_chart_dframe.groupby('Experiment'):
        all_vals.append(np.mean(group[metric_name]))
    #xloc = (max(all_vals) + xlim[0]) / 2
    xloc = xlim[0] + 0.1 * (xlim[1] - xlim[0])

    for i,m in enumerate(method_map):
        for j,method in enumerate(method_map[m]):
            if round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][0], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                lab_colors[i][j] = best_color
                font_style[i][j] = ['bold', 'italic']
            elif round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][1], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                font_style[i][j] = ['bold']
    #fig, axs = plt.subplots(ncols=1,nrows=3,figsize=(6, 17))

    #fig = plt.figure(figsize=(10,7))
    fig = plt.figure(figsize=(10,5))
    #gs = gridspec.GridSpec(len(method_map), 1, height_ratios=[0.25, 1, 1])
    gs = gridspec.GridSpec(len(method_map), 1, height_ratios=[1, 1])

    ax_prev = None
    axs = []
    for i, (g, split) in enumerate(zip(gs, [group_name_map[g] for g in group_name_map])):
        if len(axs) > 0:
            ax = plt.subplot(g, sharex=axs[0])
        else:
            ax = plt.subplot(g)
        dset = bar_chart_dframe[bar_chart_dframe['split'] == split]

        dset = dset.groupby(['Experiment', 'split'], as_index=False).agg('mean')
        if metric_name == 'F1':
            palette[i] = ["#F0988C"] + palette[i]
            if dataset[1] == 'Jigsaw':
                if split == 'KLD':
                    dset = pd.concat([pd.DataFrame([['Ind. Best', 'KLD', 59.464]], columns=['Experiment', 'split', metric_name]), dset])
                else:
                    dset = pd.concat([pd.DataFrame([['Ind. Best', 'Gold + KLD', 67.656]], columns=['Experiment', 'split', metric_name]), dset])
            else:
                if split == 'KLD':
                    dset = pd.concat(
                        [pd.DataFrame([['Ind. Best', 'KLD', 70.762]], columns=['Experiment', 'split', metric_name]), dset])
                else:
                    dset = pd.concat(
                        [pd.DataFrame([['Ind. Best', 'Gold + KLD', 72.790]], columns=['Experiment', 'split', metric_name]),
                         dset])
            best_values = list(sorted(dset[metric_name], reverse=True))
        elif metric_name == 'NLL':
            palette[i] = ["#F0988C"] + palette[i]
            if dataset[1] == 'Jigsaw':
                if split == 'KLD':
                    dset = pd.concat([pd.DataFrame([['Ind. Best', 'KLD', 0.440]], columns=['Experiment', 'split', metric_name]), dset])
                else:
                    dset = pd.concat([pd.DataFrame([['Ind. Best', 'Gold + KLD', 0.367]], columns=['Experiment', 'split', metric_name]), dset])
            else:
                if split == 'KLD':
                    dset = pd.concat(
                        [pd.DataFrame([['Ind. Best', 'KLD', 0.693]], columns=['Experiment', 'split', metric_name]), dset])
                else:
                    dset = pd.concat(
                        [pd.DataFrame([['Ind. Best', 'Gold + KLD', 0.640]], columns=['Experiment', 'split', metric_name]),
                         dset])
            best_values = list(sorted(dset[metric_name]))
        else:
            best_values = list(sorted(dset[metric_name], reverse=True))

        sns.barplot(y='Experiment', x=metric_name, hue='Experiment', data=dset, palette=palette[i], ax=ax, dodge=False)
        ax.set(xlim=xlim)
        #ax.set(xlim=(0, 100))
        for j, (container) in enumerate(ax.containers):
            #labels = ax.bar_label(container, fmt='%.3f', label_type='center', fontweight='bold')
            #for k,(bar, label) in enumerate(zip(container, labels)):
            for k,bar in enumerate(container):
                bar.set_alpha(alpha)
                if not np.isnan(bar.get_width()):
                    if bar.get_width() == best_values[0]:
                        lab_colors = best_color
                        font_style = ['bold', 'italic']
                    elif bar.get_width() == best_values[1]:
                        lab_colors = best_color
                        font_style = ['bold']
                    else:
                        lab_colors = 'black'
                        font_style = []
                    ax.annotate(f"{bar.get_width():.3f}",
                            xy=(xloc, bar.get_y() + bar.get_height() / 2),
                            ha='center', va='center', fontweight='bold' if 'bold' in font_style else 'normal',
                            color=lab_colors, fontsize=18, style='italic' if 'italic' in font_style else 'normal')

                    # ax.annotate(f"{bar.get_width():.3f}",
                    #       xy=(xloc, bar.get_y() + bar.get_height() / 2),
                    #       ha='center', va='center', fontweight='bold' if 'bold' in font_style[i][j] else 'normal',
                    #       color=lab_colors[i][j], fontsize=18, style='italic' if 'italic' in font_style[i][j] else 'normal')

        ax.legend_.remove()
        if add_y_labels:
            ax.set_ylabel(split, fontsize=24)
        else:
            ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #change_height(ax, 0.5)
        ax_prev = ax
        axs.append(ax)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    axs[0].spines['bottom'].set_visible(False)
    plt.setp(axs[1].get_xticklabels(), visible=False)
    axs[1].spines['bottom'].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    if add_y_labels:
        fig.align_ylabels(axs)
    # remove last tick label for the second subplot
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[-1].label1.set_visible(False)
    # axs[0].set_xticks(None, minor=True)
    # axs[1].set_xticks(None, minor=True)
    #plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{outfile}-bayesianonly.png")
    plt.savefig(f"{outfile}-bayesianonly.pdf", bbox_inches='tight')
    plt.close('all')


def plot_metrics(plot_dframes, metric_name, method_map, best_per_group, group_name_map, mu, outfile, dataset, add_y_labels=True):
    bar_chart_dframe = pd.DataFrame(plot_dframes[metric_name], columns=[metric_name, 'split', 'Experiment'])
    alpha = 0.65#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    lab_colors = [['black' for i in range(len(method_map[m]))] for m in method_map]
    font_style = [[[] for i in range(len(method_map[m]))] for m in method_map]
    best_color = 'black'
    #orig_palette = sns.color_palette()
    #palette = [orig_palette[0]]*4 + [orig_palette[1]] * 3 + [orig_palette[2]] * 2
    palette = [[method[2] for method in method_map[m]] for m in method_map]
    round_digits = 3
    if metric_name in ['P', 'R', 'F1']:
        # plt_min = 30
        # plt_max = 80
        xlim = dataset[2]
        #xloc = 33.5
    else:
        # plt_min = 0.3
        # plt_max = 0.9
        xlim = dataset[3]
        #xloc = 0.335

    # Get the label xloc
    all_vals = []
    for g,group in bar_chart_dframe.groupby('Experiment'):
        all_vals.append(np.mean(group[metric_name]))
    #xloc = (max(all_vals) + xlim[0]) / 2
    xloc = xlim[0] + 0.1 * (xlim[1] - xlim[0])

    for i,m in enumerate(method_map):
        for j,method in enumerate(method_map[m]):
            if round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][0], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                lab_colors[i][j] = best_color
                font_style[i][j] = ['bold', 'italic']
            elif round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][1], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                font_style[i][j] = ['bold']
    #fig, axs = plt.subplots(ncols=1,nrows=3,figsize=(6, 17))
    fig = plt.figure(figsize=(10,7))
    #fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(len(method_map), 1, height_ratios=[0.25, 1, 1])
    #gs = gridspec.GridSpec(len(method_map), 1, height_ratios=[1, 1])
    ax_prev = None
    axs = []
    for i, (g, split) in enumerate(zip(gs, [group_name_map[g] for g in group_name_map])):
        if len(axs) > 0:
            ax = plt.subplot(g, sharex=axs[0])
        else:
            ax = plt.subplot(g)
        dset = bar_chart_dframe[bar_chart_dframe['split'] == split]
        #dset = dset.groupby(['Experiment', 'split'], as_index=False).agg('mean')
        # if metric_name == 'F1':
        #     palette[i] = ["#F0988C"] + palette[i]
        #     if dataset[1] == 'Jigsaw':
        #         if split == 'KLD':
        #             dset = pd.concat([pd.DataFrame([['Ind. Best', 'KLD', 59.464]], columns=['Experiment', 'split', metric_name]), dset])
        #         else:
        #             dset = pd.concat([pd.DataFrame([['Ind. Best', 'Gold + KLD', 67.656]], columns=['Experiment', 'split', metric_name]), dset])
        #     else:
        #         if split == 'KLD':
        #             dset = pd.concat(
        #                 [pd.DataFrame([['Ind. Best', 'KLD', 68.695]], columns=['Experiment', 'split', metric_name]), dset])
        #         else:
        #             dset = pd.concat(
        #                 [pd.DataFrame([['Ind. Best', 'Gold + KLD', 71.431]], columns=['Experiment', 'split', metric_name]),
        #                  dset])
        #best_values = list(sorted(dset[metric_name], reverse=True))
        sns.barplot(y='Experiment', x=metric_name, hue='Experiment', data=dset, palette=palette[i], ax=ax, dodge=False)
        ax.set(xlim=xlim)
        #ax.set(xlim=(0, 100))
        for j, (container) in enumerate(ax.containers):
            #labels = ax.bar_label(container, fmt='%.3f', label_type='center', fontweight='bold')
            #for k,(bar, label) in enumerate(zip(container, labels)):
            for k,bar in enumerate(container):
                bar.set_alpha(alpha)
                if not np.isnan(bar.get_width()):
                #     if bar.get_width() == best_values[0]:
                #         lab_colors = best_color
                #         font_style = ['bold', 'italic']
                #     elif bar.get_width() == best_values[1]:
                #         lab_colors = best_color
                #         font_style = ['bold']
                #     else:
                #         lab_colors = 'black'
                #         font_style = []
                # ax.annotate(f"{bar.get_width():.3f}",
                #             xy=(xloc, bar.get_y() + bar.get_height() / 2),
                #             ha='center', va='center', fontweight='bold' if 'bold' in font_style else 'normal',
                #             color=lab_colors, fontsize=18, style='italic' if 'italic' in font_style else 'normal')

                    ax.annotate(f"{bar.get_width():.3f}",
                          xy=(xloc, bar.get_y() + bar.get_height() / 2),
                          ha='center', va='center', fontweight='bold' if 'bold' in font_style[i][j] else 'normal',
                          color=lab_colors[i][j], fontsize=18, style='italic' if 'italic' in font_style[i][j] else 'normal')

        ax.legend_.remove()
        if add_y_labels:
            ax.set_ylabel(split, fontsize=24)
        else:
            ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #change_height(ax, 0.5)
        ax_prev = ax
        axs.append(ax)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    axs[0].spines['bottom'].set_visible(False)
    plt.setp(axs[1].get_xticklabels(), visible=False)
    axs[1].spines['bottom'].set_visible(False)
    plt.subplots_adjust(hspace=.0)
    if add_y_labels:
        fig.align_ylabels(axs)
    # remove last tick label for the second subplot
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[-1].label1.set_visible(False)
    # axs[0].set_xticks(None, minor=True)
    # axs[1].set_xticks(None, minor=True)
    #plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{outfile}.png")
    plt.savefig(f"{outfile}.pdf", bbox_inches='tight')
    plt.close('all')


def plot_metrics2(plot_dframes, metric_name, method_map, best_per_group, group_name_map, mu, outfile, dataset, add_y_labels=True):
    bar_chart_dframe = pd.DataFrame(plot_dframes[metric_name], columns=[metric_name, 'split', 'Experiment'])
    alpha = 0.65#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    lab_colors = [['black' for i in range(len(method_map[m]))] for m in method_map]
    font_style = [[[] for i in range(len(method_map[m]))] for m in method_map]
    best_color = 'black'
    #orig_palette = sns.color_palette()
    #palette = [orig_palette[0]]*4 + [orig_palette[1]] * 3 + [orig_palette[2]] * 2
    palette = [[method[2] for method in method_map[m]] for m in method_map]
    round_digits = 3
    if metric_name in ['P', 'R', 'F1']:
        # plt_min = 30
        # plt_max = 80
        xlim = dataset[2]
        #xloc = 33.5
    else:
        # plt_min = 0.3
        # plt_max = 0.9
        xlim = dataset[3]
        #xloc = 0.335

    # Get the label xloc
    all_vals = []
    for g,group in bar_chart_dframe.groupby('Experiment'):
        all_vals.append(np.mean(group[metric_name]))
    xloc = (max(all_vals) + xlim[0]) / 2

    for i,m in enumerate(method_map):
        for j,method in enumerate(method_map[m]):
            if round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][0], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                lab_colors[i][j] = best_color
                font_style[i][j] = ['bold', 'italic']
            elif round(mu[method[0]][metric_name], round_digits) == round(best_per_group[m][1], round_digits) and method[0] not in ['kld-mu', 'mtt-mu']:
                font_style[i][j] = ['bold']
    #fig, axs = plt.subplots(ncols=1,nrows=3,figsize=(6, 17))
    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot()
    #gs = gridspec.GridSpec(len(method_map), 1, height_ratios=[0.25, 1, 1])
    dset = bar_chart_dframe.groupby(['Experiment', 'split'], as_index=False).agg('mean')
    dset = dset.pivot_table(metric_name, 'split', 'Experiment')
    dset.plot(kind='bar', ax=ax)

    plt.tight_layout()
    plt.savefig(f"{outfile}.png")
    plt.savefig(f"{outfile}.pdf", bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir",
                        help="The location of the baseline metrics",
                        type=str, required=True)
    parser.add_argument("--output_loc",
                        help="Where to save the tables",
                        type=str, default='latex/')

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_loc}"):
        os.makedirs(f"{args.output_loc}")

    #### Experiments ######
    datasets = [
        ("rte-ood-snli", "RTE", (50, 75), (0.4, 0.7)),
        ("pos-ood", "POS PTB", (60, 75), (0.55, 0.9)),
        ("jigsaw", "Jigsaw", (30, 80), (0.3, 0.6)),
        ("cifar10h", "CINIC-10", (30, 65), (0.3, 0.75)),
        #("pos-ood-conll-fixed", "POS", (60, 75), (0.65, 0.9)),
        # ("mre", "MRE-in"),
        # ("pos", "POS-in"),
        # ("rte", "RTE-in")
    ]

    methods = [
        ("majority", "Majority", "#000000"),
        ("kld-standard", "Standard", "#F0988C"),
        ("kld-softmax", "Softmax", "#F0988C"),
        ("kld-ds", "DS", "#F0988C"),
        ("kld-mace", "MACE", "#F0988C"),
        ("kld-glad", "GLAD", "#F0988C"),
        ("kld-wawa", "WaWA", "#F0988C"),
        ("kld-zbs", "ZBS", "#F0988C"),
        ("kld-ensemble_basic", "Agg", '#A1A9D0'),
       ]

    metric_map = {
        "F1": ("F1", "max"),
        "test_NLL_post": ("NLL", "min"),
        #"test_Brier_post": ("Brier score", "min")
    }

    # ylabel_datasets = ['RTE', 'POS PTB']
    # #ylabel_datasets = ['Jigsaw']


    def format_string(mu, std, metric, method):
        if metric in ["test_NLL_post", "test_Brier_post"]:
            return f"{mu[method[0]][metric_map[metric][0]]:.3f}_{{{std[method[0]][metric_map[metric][0]]:.2f}}}"
        else:
            return f"{mu[method[0]][metric_map[metric][0]]:.2f}_{{{std[method[0]][metric_map[metric][0]]:.2f}}}"

    # method_to_ranks_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list), 'test_Brier_post': defaultdict(list)}
    # all_score_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list), 'test_Brier_post': defaultdict(list)}
    method_to_ranks_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list),
                           }
    all_score_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list)}
    rank_map_combined = defaultdict(list)
    table_head = METRICS_TABLE_HEAD
    table_row_strings = [''] * len(methods)
    for n,dataset in enumerate(datasets):
        table_head = table_head.replace(f"{{d{n}}}", dataset[1])
        if not os.path.exists(f"{args.output_loc}/{dataset[1]}"):
            os.makedirs(f"{args.output_loc}/{dataset[1]}")
        plot_dframes = defaultdict(list)
        # Iterate through all metrics files for ranking baseline
        # This is the main table for training and testing on ALL of the data; test is on
        # All data, then broken down into tweets and news
        metrics_by_baseline = defaultdict(lambda: {'F1': [],
                                                   'NLL': [], 'Brier score': []})

        for method in methods:

            for fname in (Path(args.metrics_dir)/dataset[0]/method[0]).glob('**/*.json'):
                with open(fname) as f:
                    metrics = json.loads(f.read())
                for m in metric_map:
                    if m in ["test_NLL_post", "test_Brier_post"]:
                        metrics_by_baseline[method[0]][metric_map[m][0]].append(metrics[m])
                    else:
                        metrics_by_baseline[method[0]][metric_map[m][0]].append(metrics[m] * 100)
                    #plot_dframes[metric_map[m][0]].append([metrics_by_baseline[method[0]][metric_map[m][0]][-1], group_name_map[group], method[1]])

        # Get the means and variances
        mu = defaultdict(dict)
        std = defaultdict(dict)
        metrics_str = ''
        for method in methods:
            for m in metric_map:
                # if len(metrics_by_baseline[method[0]][metric_map[m][0]]) == 0:
                #     ipdb.set_trace()
                mu[method[0]][metric_map[m][0]] = np.array(metrics_by_baseline[method[0]][metric_map[m][0]]).mean()
                std[method[0]][metric_map[m][0]] = np.array(metrics_by_baseline[method[0]][metric_map[m][0]]).std()

        # Calculate mean reciprocal ranks for each method
        #for metric in ['F1', 'test_NLL_post', 'test_Brier_post']:
        for metric in ['F1', 'test_NLL_post']:
            # Collect all the F1 scores for this group
            #name_to_score = [[method[1], mu[method[0]][metric_map[metric][0]]] for method in methods[group]]
            name_to_score = {method[1]: metrics_by_baseline[method[0]][metric_map[metric][0]] for method in methods}
            name_index = list(name_to_score.keys())
            for method in name_to_score:
                all_score_map[metric][method].extend(name_to_score[method])

            value_mat = np.vstack([np.array(v) for v in name_to_score.values()])
            orders = np.argsort(value_mat,0)
            if metric == 'F1':
                orders = orders[::-1,:]
            for j,name_i in enumerate(name_index):
                for k, ranking in enumerate(orders.T):
                    init = [0] * len(name_index)
                    init[j] = 1
                    method_to_ranks_map[metric][name_i].append(np.array(init)[ranking])
                    rank_map_combined[name_i].append(method_to_ranks_map[metric][name_i][-1])
            # for i,row in enumerate(name_to_score):
            #     init = [0] * len(name_to_score)
            #     init[i] = 1
            #     method_to_ranks_map[group][metric][row[0]].append(np.array(init)[order])




        table_string = METRICS_TABLE_HEAD + '\n'
        table_string += "\\midrule\n"
        method_names = [method[0] for method in methods]
        # Rank the metrics
        ranks = {}
        for m in metric_map:
            values = [(method[0], mu[method[0]][metric_map[m][0]]) for method in methods]
            ranks[metric_map[m][0]] = list(sorted(values, key=lambda x: x[1], reverse=metric_map[m][1] == "max"))

        for i,method in enumerate(methods):
            if table_row_strings[i] == '':
                table_row_strings[i] += f"{method[1]}&"
            for m in metric_map:
                if ranks[metric_map[m][0]][0][0] == method[0]:
                    table_row_strings[i] += f"$\\mathbf{{{format_string(mu, std, m, method)}}}$& "
                # elif ranks[metric_map[m][0]][1][0] == method[0]:
                #     table_string += f"$\\underline{{{format_string(mu, std, m, method)}}}$& "
                else:
                    table_row_strings[i] += f"${format_string(mu, std, m, method)}$& "

    table_string = table_head + '\n'
    table_string += "\\midrule\n"
    for row_string in table_row_strings:
        table_string += row_string[:-2] + "\\\\\n"

    table_string += METRICS_TABLE_FOOT

    with open(f"{args.output_loc}/metrics.tex", 'wt') as f:
        f.write(table_string)
