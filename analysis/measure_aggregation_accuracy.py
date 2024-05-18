import argparse
import sys
sys.path.insert(0,'.')
import os

import numpy as np
import pandas as pd
from util.datareader import aggregate, get_crowdkit_distributions, get_annotation_matrix, POS_LABEL_MAP, normalized_distribution, softmax_distribution, crowdkit_classes
from scipy.special import softmax
from scipy.stats import entropy
import ipdb
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from util.temperature_scaling import JSD
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from util.datareader import get_original_jigsaw_distribution
import torchvision


METRICS_TABLE_HEAD = """
\\begin{table}%[htp]
    \\def\\arraystretch{1.2}
    \\centering
    \\fontsize{10}{10}\\selectfont
    \\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c c c}
    \\toprule %\\thickhline
    Method & RTE & POS & Jigsaw & CIFAR-10H \\\\
    \\midrule
"""

METRICS_TABLE_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{The accuracy of each annotation method with respect to the expert annotations in each dataset.}
    \\label{tab:dataset_accuracy}
\\end{table}
"""

METRICS_CALIBRATION_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Negative log likelihood of each annotation method with respect to the expert annotations in each dataset.}
    \\label{tab:dataset_accuracy}
\\end{table}
"""

METRICS_ENTROPY_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Average entropy of soft labels for each annotation method.}
    \\label{tab:dataset_accuracy}
\\end{table}
"""


def read_fornaciarni_pos_transformers(data_dir, seed=1000, distributions=list(crowdkit_classes.keys()) + ['standard', 'softmax']):

    data = pd.read_excel(f"{data_dir}/dh.xlsx")
    data['word'] = data['word'].astype(str)
    gold_ids = list(data['id_word'])

    train_data = data[data['set'] == 'trn']
    dev_data = data[data['set'] == 'dev']
    test_data = data[data['set'] == 'hard-tst']

    if os.path.exists(f"{data_dir}/formatted_annotations.csv"):
        annotations_df = pd.read_csv(f"{data_dir}/formatted_annotations.csv")
    else:
        data = pd.read_csv(f"{data_dir}/gimpel_crowdsourced-tf.tsv", sep='\t', header=None)
        annotations = []
        for i, row in data.iterrows():
            for j,lab in enumerate(row[2].split(',')):
                if lab != '':
                    annotations.append([i, j, POS_LABEL_MAP[lab]])
        annotations_df = pd.DataFrame(annotations, columns=['task', 'worker', 'label'])
        annotations_df.to_csv(f"{data_dir}/formatted_annotations.csv", index=None)

    annotations,_ = get_annotation_matrix(annotations_df, {i:i for i in range(12)})

    # First get standard and softmax distributions
    dist_dict = {}
    if 'softmax' in distributions:
        dist_dict['softmax'] = softmax_distribution(annotations, {i:i for i in range(12)})
    if 'standard' in distributions:
        dist_dict['standard'] = normalized_distribution(annotations, {i:i for i in range(12)})


    dist_dict = get_crowdkit_distributions(data_dir, dist_dict, annotations_df, distributions=distributions, seed=seed)

    # Now get them into a matrix instead of dict
    for dist in dist_dict:
        dist_dict[dist] = np.array([dist_dict[dist][task] for task in range(len(train_data) + len(dev_data))])

    dist_dict['ensemble_basic'] = aggregate(dist_dict, 'ensemble_basic')

    return np.array([int(v) for v in train_data['gold']] + [int(v) for v in dev_data['gold']]), dist_dict


def read_rte_transformers(data_dir, seed=1000, distributions=list(crowdkit_classes.keys()) + ['standard', 'softmax']):
    rte_raw = pd.read_csv(f"{data_dir}/rte.standardized.tsv", sep='\t')
    rte_raw = rte_raw.rename({'!amt_worker_ids': 'worker', 'orig_id': 'task', 'response': 'label'}, axis=1)
    orig_sents = pd.read_csv(f"{data_dir}/rte1.tsv", sep='\t')
    # Drop and rename columns
    orig_map = {row['id']: i for i,row in orig_sents.iterrows()}

    selection = []
    for g, group in rte_raw.groupby('task'):
        selection.append(orig_map[g])

    dataset = orig_sents.iloc[selection]

    annotations, _ = get_annotation_matrix(rte_raw, {i: i for i in range(2)})

    # First get standard and softmax distributions
    dist_dict = {}
    if 'softmax' in distributions:
        dist_dict['softmax'] = softmax_distribution(annotations, {i: i for i in range(2)})
    if 'standard' in distributions:
        dist_dict['standard'] = normalized_distribution(annotations, {i: i for i in range(2)})

    dist_dict = get_crowdkit_distributions(data_dir, dist_dict, rte_raw, distributions=distributions, seed=seed)

    # Now get them into a matrix instead of dict
    for dist in dist_dict:
        dist_dict[dist] = np.array([dist_dict[dist][row['id']] for i,row in dataset.iterrows()])


    dist_dict['ensemble_basic'] = aggregate(dist_dict, 'ensemble_basic')


    return np.array([int(v) for v in dataset['value']]), dist_dict


def read_jigsaw_data_transformers(data_dir, seed=1000, distributions=list(crowdkit_classes.keys()) + ['standard', 'softmax']):
    # Collapse "very toxic" and "toxic" into one class and "not sure" and "not toxic" into another
    # The gold targets are the average number of annotations rated "not sure" and "not toxic",
    # so anything > 0.5 becomes class 1 and anything <= 0.5 becomes class 0
    label_map = {-2: 1, -1: 1, 0: 0, 1: 0}
    jigsaw_raw = pd.read_csv(f"{data_dir}/specialized_rater_pools_data.csv").fillna('')
    dataset = pd.read_csv(f"{data_dir}/gold.csv").fillna('')

    if os.path.exists(f"{data_dir}/formatted_annotations.csv"):
        annotations_df = pd.read_csv(f"{data_dir}/formatted_annotations.csv")
    else:
        annotations = []
        for g, group in jigsaw_raw.groupby('id'):
            for j, row in group.iterrows():
                if row['toxic_score'] != '':
                    annotations.append([g, row['unique_contributor_id'], label_map[row['toxic_score']]])
        annotations_df = pd.DataFrame(annotations, columns=['task', 'worker', 'label'])
        annotations_df.to_csv(f"{data_dir}/formatted_annotations.csv", index=None)

    annotations, _ = get_annotation_matrix(annotations_df, {i: i for i in range(2)})

    # First get standard and softmax distributions
    dist_dict = {}
    if 'softmax' in distributions:
        dist_dict['softmax'] = softmax_distribution(annotations, {i: i for i in range(2)})
    if 'standard' in distributions:
        dist_dict['standard'] = normalized_distribution(annotations, {i: i for i in range(2)})

    dist_dict = get_crowdkit_distributions(data_dir, dist_dict, annotations_df, seed=seed, distributions=distributions)

    # Now get them into a matrix instead of dict
    for dist in dist_dict:
        dist_dict[dist] = np.array([dist_dict[dist][task] for task in dataset['id']])

    dist_dict['ensemble_basic'] = aggregate(dist_dict, 'ensemble_basic')

    return np.array([int(v) for v in dataset['gold']]), dist_dict


def read_cifar10h_transformers(data_dir, seed=1000, distributions=list(crowdkit_classes.keys()) + ['standard', 'softmax']):
    cifar_raw = pd.read_csv(f"{data_dir}/cifar-10h/data/cifar10h-raw.csv").fillna('')
    cifar_raw = cifar_raw[cifar_raw['cifar10_test_test_idx'] >= 0]
    dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar-10h/data/', train=False,
                                             download=True)

    if os.path.exists(f"{data_dir}/formatted_annotations.csv"):
        annotations_df = pd.read_csv(f"{data_dir}/formatted_annotations.csv")
    else:
        annotations = {}
        worker_to_colno = {}
        for i, (g, _) in enumerate(cifar_raw.groupby('annotator_id')):
            worker_to_colno[g] = i

        for g, group in cifar_raw.groupby('cifar10_test_test_idx'):
            annotations[g] = {}
            for j, row in group.iterrows():
                if row['chosen_label'] != '':
                    annotations[g][worker_to_colno[row['annotator_id']]] = int(row['chosen_label'])

        final_annotations = []
        for id_ in annotations:
            for worker in annotations[id_]:
                final_annotations.append([id_, worker, annotations[id_][worker]])
        annotations_df = pd.DataFrame(final_annotations, columns=['task', 'worker', 'label'])
        annotations_df.to_csv(f"{data_dir}/formatted_annotations.csv", index=None)


    n_labels = len(set(annotations_df['label']))
    annotations, _ = get_annotation_matrix(annotations_df, {i: i for i in range(n_labels)})

    # First get standard and softmax distributions
    dist_dict = {}
    if 'softmax' in distributions:
        dist_dict['softmax'] = softmax_distribution(annotations, {i: i for i in range(n_labels)})
    if 'standard' in distributions:
        dist_dict['standard'] = normalized_distribution(annotations, {i: i for i in range(n_labels)})

    dist_dict = get_crowdkit_distributions(data_dir, dist_dict, annotations_df, distributions=distributions, seed=seed)

    # Now get them into a matrix instead of dict
    for dist in dist_dict:
        dist_dict[dist] = np.array([dist_dict[dist][task] for task in range(len(dataset))])

    dist_dict['ensemble_basic'] = aggregate(dist_dict, 'ensemble_basic')

    return np.array([int(v[1]) for v in dataset]), dist_dict


if __name__ == '__main__':

    def print_all_accs(gold, distributions, m_a_map, m_c_map, m_e_map):
        for dist in distributions:
            preds = np.argmax(distributions[dist], -1)
            print(f"{dist}: {(preds == gold).astype(np.int32).mean()}")
            # Apply label smoothing for nll
            n_labels = distributions[dist].shape[1]
            eps = 0.01
            distributions[dist] = (1 - eps)*distributions[dist] + eps / n_labels
            nll = F.nll_loss(torch.tensor(distributions[dist]).log(), torch.tensor(gold), reduction='mean').item()
            print(f"{dist} (nll): {nll}")
            m_a_map[dist].append((preds == gold).astype(np.int32).mean())
            m_c_map[dist].append(nll)
            m_e_map[dist].append(entropy(distributions[dist], axis=-1).mean())

    method_accuracy_map = defaultdict(list)
    method_calibration_map = defaultdict(list)
    method_entropy_map = defaultdict(list)
    # Generate some heatmaps with JSD
    gold, rte_dist_dict = read_rte_transformers('./data/rte')
    print("RTE")
    print("==================================")
    print_all_accs(gold, rte_dist_dict,
        method_accuracy_map, method_calibration_map, method_entropy_map)
    print("\n\n")

    gold, pos_dist_dict = read_fornaciarni_pos_transformers('data/pos')
    print("POS")
    print("==================================")
    print_all_accs(gold, pos_dist_dict,
        method_accuracy_map, method_calibration_map, method_entropy_map)
    print("\n\n")

    gold, jigsaw_dist_dict = read_jigsaw_data_transformers('./data/jigsaw')
    print("Jigsaw")
    print("==================================")
    print_all_accs(gold, jigsaw_dist_dict,
        method_accuracy_map, method_calibration_map, method_entropy_map)
    print("\n\n")

    gold, cifar_dist_dict = read_cifar10h_transformers('./data/cifar10h')
    print("Jigsaw")
    print("==================================")
    print_all_accs(gold, cifar_dist_dict,
                   method_accuracy_map, method_calibration_map, method_entropy_map)
    print("\n\n")

    # Write out latex tables
    latex_string = METRICS_TABLE_HEAD + '\n'
    row_name_map = {
        'standard': 'Standard',
        'softmax': 'Softmax',
        'mace': 'MACE',
        'ds': 'DS',
        'glad': "GLAD",
        'wawa': 'WaWA',
        'zbs': 'ZBS',
        'ensemble_basic': 'Agg'
    }
    for method in row_name_map:
        latex_string += f"{row_name_map[method]}"
        for val in method_accuracy_map[method]:
            latex_string += f" & {val*100:.2f} "
        latex_string += '\\\\\n'
    latex_string += METRICS_TABLE_FOOT

    with open('./latex/method_accuracy.tex', 'wt') as f:
        f.write(latex_string)

    latex_string = METRICS_TABLE_HEAD + '\n'
    for method in row_name_map:
        latex_string += f"{row_name_map[method]}"
        for val in method_calibration_map[method]:
            latex_string += f" & {val:.3f} "
        latex_string += '\\\\\n'
    latex_string += METRICS_CALIBRATION_FOOT

    with open('./latex/method_calibration.tex', 'wt') as f:
        f.write(latex_string)

    latex_string = METRICS_TABLE_HEAD + '\n'
    for method in row_name_map:
        latex_string += f"{row_name_map[method]}"
        for val in method_entropy_map[method]:
            latex_string += f" & {val:.2f} "
        latex_string += '\\\\\n'
    latex_string += METRICS_ENTROPY_FOOT

    with open('./latex/method_entropy.tex', 'wt') as f:
        f.write(latex_string)