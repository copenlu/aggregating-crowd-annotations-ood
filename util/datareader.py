import os
import torchvision
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import arviz as az
from jax import nn, random, vmap
from functools import partial
import ipdb
from scipy.special import softmax
from scipy.stats import entropy
from transformers import DataCollatorForTokenClassification
import torch
from collections import defaultdict
import nltk
from nltk.tag.mapping import map_tag
from copy import deepcopy
from crowdkit.aggregation import (
    DawidSkene,
    GLAD,
    MACE,
    Wawa,
    ZeroBasedSkill
)

from util.temperature_scaling import calculate_temperature_scaled_ensemble_multiple
from util.temperature_scaling import calculate_temperature_scaled_ensemble_maximize
from util.temperature_scaling import calculate_jensen_shannon_centroid
from util.temperature_scaling import temperature_scale_distribution



jax.config.update('jax_platform_name', 'cpu')


POS_LABEL_MAP = {'': -1,
                 'ADJ': 0,
                 'ADP': 1,
                 'ADV': 2,
                 'CONJ': 3,
                 'DET': 4,
                 'NOUN': 5,
                 'NUM': 6,
                 'PRON': 7,
                 'PRT': 8,
                 '.': 9,
                 'VERB': 10,
                 'X': 11,
                 '[CLS]': 12,
                 '[SEP]': 13,
                 '[SUB]': 14,
                 '[PAD]': 15
                 }

crowdkit_classes = {
    'ds': DawidSkene,
    'glad': GLAD,
    'mace': MACE,
    'wawa': Wawa,
    'zbs': ZeroBasedSkill
}


class DataCollatorForTokenClassificationBayes(DataCollatorForTokenClassification):

    label_mixer: bool= False

    def torch_call(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        majority_labels = [feature['majority_labels'] for feature in features] if 'majority_labels' in features[0].keys() else None
        bayes = [feature['bayes'] for feature in features] if 'bayes' in features[0].keys() else None
        idx = [feature['idx'] for feature in features] if 'idx' in features[0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        batch['idx'] = idx
        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            if majority_labels:
                batch['majority_labels'] = [
                    list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in majority_labels
                ]
            if bayes:
                if self.label_mixer:
                    batch['bayes'] = [[list(b) + [[0.0] * (len(b[0]) - 1) + [1.0]] * (sequence_length - len(b)) for b in dist] for dist in bayes]
                else:
                    batch['bayes'] = [list(b) + [[0.0] * (len(b[0]) - 1) + [1.0]] * (sequence_length - len(b)) for b in bayes]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]
            if majority_labels:
                batch['majority_labels'] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in majority_labels
                ]
            if bayes:
                if self.label_mixer:
                    batch['bayes'] = [[[[0.0] * (len(b[0]) - 1) + [1.0]] * (sequence_length - len(b)) + list(b) for b in dist] for dist in
                                      bayes]
                else:
                    batch['bayes'] = [[[0.0] * (len(b[0]) - 1) + [1.0]] * (sequence_length - len(b)) + list(b) for b in bayes]

        try:
            torch.tensor(batch['bayes'])
        except:
            ipdb.set_trace()
        batch = {k: torch.tensor(v, dtype=torch.int64 if k != 'bayes' else torch.float32) for k, v in batch.items()}
        return batch


def aggregate(distribution_dict, aggregation='bayes', lam=1e-3, lr=1e-2, weights=None):
    if weights is None:
        distributions = [torch.tensor(v) for k,v in distribution_dict.items()]
    else:
        distributions = [torch.tensor(distribution_dict[k]) for k in weights]
        weights = [weights[k] for k in weights]

    if aggregation == 'ensemble_basic':
        return np.average(torch.stack(distributions).numpy(), weights=weights, axis=0)

        #return (bayes_dist + softmax_dist + standard_dist + mace_dist) / 4
    elif aggregation == 'ensemble_conflation':
        # Method which favors high entropy distributions
        product = np.prod(torch.stack(distributions).numpy(), axis=0)
        return product / product.sum(-1, keepdims=True)

    elif aggregation == 'ensemble_temperature':
        class_dist, T = calculate_temperature_scaled_ensemble_multiple(distributions,
                                                                        lr=lr)
        print(T)
        return np.array(class_dist)
    elif aggregation == 'ensemble_temperature_maximize':
        class_dist, T = calculate_temperature_scaled_ensemble_maximize(distributions,
                                                                        lr=lr)
        print(T)
        return np.array(class_dist)
    elif aggregation == 'ensemble_centroid':
        class_dist = calculate_jensen_shannon_centroid(distributions)
        #print(T)
        return np.array(class_dist)
    elif aggregation == 'ensemble_hybrid':
        # Get temperatures for the different distributions
        _, T = calculate_temperature_scaled_ensemble_multiple(distributions,
                                                               lr=lr)
        print(T)
        # Calculate the JSC
        class_dist = calculate_jensen_shannon_centroid([temperature_scale_distribution(distributions[i], T[i]).detach().numpy() for i in range(len(distributions))])
        return class_dist
    elif aggregation == 'ensemble_hybrid_maximize':
        # Get temperatures for the different distributions
        _, T = calculate_temperature_scaled_ensemble_maximize(distributions,
                                                               lr=lr)
        print(T)
        # Calculate the JSC
        class_dist = calculate_jensen_shannon_centroid([temperature_scale_distribution(distributions[i], T[i]).detach().numpy() for i in range(len(distributions))])
        return class_dist
    elif aggregation == 'label_mixer':
        return list(distribution_dict.values())
    else:
        return distribution_dict[aggregation]


def get_annotation_matrix(annotations_table, label_map=None):
    annotations = {}
    worker_to_colno = {}
    n = 0
    for i, (g, _) in enumerate(annotations_table.groupby('worker')):
        worker_to_colno[g] = i
    for g, group in annotations_table.groupby('task'):
        annotations[g] = ([-1] * len(worker_to_colno))
        for j, row in group.iterrows():
            if row['label'] not in label_map:
                label_map[row['label']] = n
                n += 1
            annotations[g][worker_to_colno[row['worker']]] = label_map[row['label']]
    return annotations, label_map


def normalized_distribution(annotations, label_map):
    """
    Return a distribution which is simply normalized over the number of annotations
    :param annotations:
    :return:
    """
    # Get standard normalized distribution for all instances
    mat = np.array([annotations[k] for k in annotations])
    # Get counts for each label
    class_logits = np.array([np.bincount(row[row > -1], minlength=len(label_map)) for row in mat])
    dist = class_logits / class_logits.sum(-1, keepdims=True)

    return {k: list(dist[i]) for i,k in enumerate(annotations)}


def softmax_distribution(annotations, label_map):
    """
    Return a distribution which is a softmax over the annotations
    :param annotations:
    :return:
    """
    # Get standard normalized distribution for all instances
    mat = np.array([annotations[k] for k in annotations])
    # Get counts for each label
    class_logits = np.array([np.bincount(row[row > -1], minlength=len(label_map)) for row in mat])
    dist = softmax(class_logits, axis=-1)

    return {k: list(dist[i]) for i, k in enumerate(annotations)}


def read_mace_distribution(mace_dir, question_order):
    # Get the predictions
    mace_preds = pd.read_csv(f"{mace_dir}/predictions.tsv", sep='\t', header=None)
    # Get the row to question mapping
    row_to_q = pd.read_csv(f"{mace_dir}/row_to_q.csv", header=None)
    row_to_q = {row[0]: row[1] for i,row in row_to_q.iterrows()}

    # Get proper distributions, map to questions
    q_to_dist = {}
    for i,row in mace_preds.iterrows():
        lab_value = []
        for key, item in row.iteritems():
            lab,value = item.split()
            lab_value.append((int(lab), float(value)))
        # sort
        row_dist = [lv[1] for lv in sorted(lab_value, key=lambda x: x[0])]
        q_to_dist[row_to_q[i]] = row_dist
    # Get them in the correct order and return
    return [q_to_dist[q] for q in question_order]


def preprocess_pos_data(tk, examples):
    # Need to add extra classes for [cls], [sep], and subword
    examples['idx'] = []
    examples['input_ids'] = []
    examples['labels'] = []
    examples['majority_labels'] = []
    examples['bayes'] = []
    examples['attention_mask'] = []
    n_bayes = 1 if len(np.array(examples['bayes_dist'][0]).shape) == 2 else np.array(examples['bayes_dist'][0]).shape[1]
    # Get the tokenized sentence
    for i in range(len(examples['word'])):
        #tokens = tk(examples['word'][i], add_special_tokens=False)
        # Align labels and combine
        out_tokens = [tk.cls_token_id]
        out_labels = [POS_LABEL_MAP['[CLS]']]
        out_majority_labels = [POS_LABEL_MAP['[CLS]']]
        out_bayes = [[[0.0] * (len(POS_LABEL_MAP) - 1)] for _ in range(n_bayes)]
        for b in out_bayes:
            b[-1][POS_LABEL_MAP['[CLS]']] = 1.0
        sub_bayes = [0.0] * (len(POS_LABEL_MAP) - 1)
        sub_bayes[POS_LABEL_MAP['[SUB]']] = 1.0
        for token,gold,majority,bayes in zip(examples['word'][i], examples['gold'][i], examples['majority_label'][i], examples['bayes_dist'][i]):
            tok = tk(token, add_special_tokens=False)
            out_tokens.extend(tok['input_ids'])
            out_labels.extend([int(gold)] + [int(POS_LABEL_MAP['[SUB]'])] * (len(tok['input_ids']) - 1))
            out_majority_labels.extend([majority] + [int(POS_LABEL_MAP['[SUB]'])] * (len(tok['input_ids']) - 1))
            for j in range(n_bayes):
                if n_bayes > 1:
                    out_bayes[j].extend([bayes[j]] + [sub_bayes] * (len(tok['input_ids']) - 1))
                else:
                    out_bayes[j].extend([bayes] + [sub_bayes] * (len(tok['input_ids']) - 1))

        out_tokens.append(tk.sep_token_id)
        out_labels.append(POS_LABEL_MAP['[SEP]'])
        out_majority_labels.append(POS_LABEL_MAP['[SEP]'])
        for j in range(n_bayes):
            out_bayes[j].append([0.0] * (len(POS_LABEL_MAP) - 1))
            out_bayes[j][-1][POS_LABEL_MAP['[SEP]']] = 1.0

        examples['input_ids'].append(out_tokens)
        examples['labels'].append(out_labels)
        examples['majority_labels'].append(out_majority_labels)
        examples['bayes'].append(out_bayes[0] if n_bayes == 1 else out_bayes)
        examples['attention_mask'].append([1]*len(out_tokens))
        examples['idx'].append(i)
    return examples


def preprocess_rte_data(tk, examples):
    batch = tk(examples['text'], text_pair=examples['hypothesis'])
    batch['labels'] = [int(v) for v in examples['value']]
    batch['majority_labels'] = examples['majority_label']
    batch['bayes'] = examples['bayes_dist']

    return batch


def preprocess_mre_data(tk, examples):
    # Mark the entities
    sentences = []
    for b1,e1,b2,e2,sentence in zip(
            examples['b1'],
            examples['e1'],
            examples['b2'],
            examples['e2'],
            examples['sentence']
    ):
        # to_insert = {
        #     '<e1>': b1, '</e1>': e1, '<e2>': b2, '</e2>': e2
        # }
        # offset = 0
        # for k,v in sorted(to_insert.items(), key=lambda x: x[1]):
        #     sentence = sentence[:v+offset] + k + sentence[v+offset:]
        #     offset += len(k)
        sentences.append(sentence)
        # if b1 < b2:
        #     assert e1 < e2, "weird overlapping spans"
        #     sentences.append(sentence[:b1] + '<e1>' + sentence[b1:e1] +
        #                      '</e1>' + sentence[e1:b2] + '<e2>' + sentence[b2:e2] + '</e2>' + sentence[e2:])
        # else:
        #     assert e2 < e1, "weird overlapping spans"
        #     sentences.append(sentence[:b2] + '<e2>' + sentence[b2:e2] +
        #                      '</e2>' + sentence[e2:b1] + '<e1>' + sentence[b1:e1] + '</e1>' + sentence[e1:])
    batch = tk(sentences)
    batch['labels'] = [int(v) for v in examples['gold']]
    batch['majority_labels'] = examples['majority_label']
    batch['bayes'] = examples['bayes_dist']

    return batch


def preprocess_jigsaw_data(tk, examples):
    batch = tk(examples['comment_text'], truncation='longest_first')
    batch['labels'] = [int(v) for v in examples['gold']]
    batch['majority_labels'] = examples['majority_label']
    batch['bayes'] = examples['bayes_dist']
    batch['soft_gold'] = examples['target']

    return batch


def get_crowdkit_distributions(data_dir, dist_dict, annotations, distributions=list(crowdkit_classes.keys()), seed=1000):
    # Get get all of the crowdkit annotations
    for method in distributions:
        if method in crowdkit_classes:
            if os.path.exists(f"{data_dir}/{method}_{seed}.csv"):
                method_df = pd.read_csv(f"{data_dir}/{method}_{seed}.csv", index_col=0)
            else:
                print(f"Calculating {method}...")
                if method == 'mace':
                    method_df = crowdkit_classes[method](random_state=seed).fit_predict_proba(annotations)
                else:
                    method_df = crowdkit_classes[method]().fit_predict_proba(annotations)
                # Sort the classes
                method_df = method_df.reindex(sorted(method_df.columns), axis=1)
                method_df.to_csv(f"{data_dir}/{method}_{seed}.csv")
            # turn into dict
            method_dict = {}
            for i, row in method_df.iterrows():
                method_dict[i] = list(row.values)
            dist_dict[method] = method_dict

    return dist_dict


def read_fornaciarni_pos_transformers(data_dir, tokenizer, aggregation, weights=None, print_accuracy=False, seed=1000, distributions=list(crowdkit_classes.keys())):

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

    datasets = []
    if aggregation != 'all_individual':
        class_dist = aggregate(dist_dict,
                                aggregation,
                                lr=1e-3,
                               weights=weights)
        # class_dist = aggregate({'bayes': bayes_dist,
        #                         'mace': mace_dist},
        #                        aggregation,
        #                        lr=1e-3)
        if aggregation != 'label_mixer':
            class_dist = np.hstack([class_dist, np.array([[0,0,0,0]] * len(class_dist))])
        else:
            class_dist = np.stack([np.hstack([dist, np.array([[0,0,0,0]] * len(dist))]) for dist in class_dist], 1)
        if print_accuracy:
            gold = np.array([int(lab) for lab in train_data['gold'].tolist() + dev_data['gold'].tolist()])
            for dist in dist_dict:
                pred = np.argmax(dist_dict[dist], -1)
                print(f"Accuracy of {aggregation}: {np.mean(pred == gold)}")
            pred = np.argmax(class_dist, -1)
            print(f"Accuracy {aggregation}: {np.mean(pred == gold)}")
        # Get distribution
        #class_dist = class_dist / class_dist.sum(-1, keepdims=True)
        # TODO: Line up all counts with rows in train and validation data
        n_train = len(train_data)
        train_data.insert(len(train_data.columns), 'bayes_dist', class_dist[:n_train].tolist())
        if aggregation == 'label_mixer':
            train_data.insert(len(train_data.columns), 'majority_label', np.argmax(dist_dict['softmax'][:n_train], -1).tolist())
        else:
            train_data.insert(len(train_data.columns), 'majority_label', np.argmax(class_dist[:n_train], -1).tolist())
        train_data = train_data[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent', as_index=False).agg({'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

        dev_data.insert(len(dev_data.columns), 'bayes_dist', class_dist[n_train:].tolist())
        if aggregation == 'label_mixer':
            dev_data.insert(len(dev_data.columns), 'majority_label', np.argmax(dist_dict['softmax'][n_train:], -1).tolist())
        else:
            dev_data.insert(len(dev_data.columns), 'majority_label', np.argmax(class_dist[n_train:], -1).tolist())
        dev_data = dev_data[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent', as_index=False).agg({'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

        test_data.insert(len(test_data.columns), 'bayes_dist', [[0.]*len(class_dist[0])] * len(test_data))
        test_data.insert(len(test_data.columns), 'majority_label', [0]*len(test_data))
        test_data = test_data[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent', as_index=False).agg({'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

        dataset = DatasetDict()
        dataset['train'] = Dataset.from_pandas(train_data)
        dataset['validation'] = Dataset.from_pandas(dev_data)
        dataset['test'] = Dataset.from_pandas(test_data)

        dataset = dataset.map(partial(preprocess_pos_data, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
        return dataset
    else:
        datasets = {}
        for method in dist_dict:
            class_dist = np.hstack([dist_dict[method], np.array([[0, 0, 0, 0]] * len(dist_dict[method]))])

            # Get distribution
            # class_dist = class_dist / class_dist.sum(-1, keepdims=True)
            # TODO: Line up all counts with rows in train and validation data
            train_data_curr = deepcopy(train_data)
            n_train = len(train_data_curr)
            train_data_curr['bayes_dist'] = class_dist[:n_train].tolist()
            train_data_curr['majority_label'] = np.argmax(class_dist[:n_train], -1).tolist()
            train_data_curr = train_data_curr[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent',
                                                                                                         as_index=False).agg(
                {'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

            dev_data_curr = deepcopy(dev_data)
            dev_data_curr['bayes_dist'] = class_dist[n_train:].tolist()
            dev_data_curr['majority_label'] = np.argmax(class_dist[n_train:], -1).tolist()
            dev_data_curr = dev_data_curr[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent',
                                                                                                     as_index=False).agg(
                {'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

            test_data_curr = deepcopy(test_data)
            test_data_curr['bayes_dist'] = [[0.] * len(class_dist[0])] * len(test_data_curr)
            test_data_curr['majority_label'] = [0] * len(test_data_curr)
            test_data_curr = test_data_curr[['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']].groupby('id_sent',
                                                                                                       as_index=False).agg(
                {'word': list, 'gold': list, 'majority_label': list, 'bayes_dist': list})

            dataset = DatasetDict()
            dataset['train'] = Dataset.from_pandas(train_data_curr)
            dataset['validation'] = Dataset.from_pandas(dev_data_curr)
            dataset['test'] = Dataset.from_pandas(test_data_curr)

            datasets[method] = dataset.map(partial(preprocess_pos_data, tokenizer), batched=True,
                                  remove_columns=dataset['train'].column_names)

        return datasets


def read_penn_treebank_transformers(tokenizer=None):
    # Get the sentences/tags
    ptb_data_raw = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
    data = []
    for i,sent in enumerate(ptb_data_raw):
        data.append([
            [w[0] for w in sent],
            i,
            [POS_LABEL_MAP[w[1]] for w in sent],
            [POS_LABEL_MAP[w[1]] for w in sent],
            [[0.]*(len(POS_LABEL_MAP) - 1)]*len(sent)
        ])
    dataset = Dataset.from_pandas(pd.DataFrame(data, columns=['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']))
    if tokenizer:
        dataset = dataset.map(partial(preprocess_pos_data, tokenizer), batched=True, remove_columns=dataset.column_names)

    return dataset


def read_conll03_transformers(tokenizer):
    conll03_dataset = load_dataset('conll2003')
    # Need to remap the tags in the test set
    tag_map = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11,
               'DT': 12, 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
               'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
               'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
               'WP': 44, 'WP$': 45, 'WRB': 46}
    int_to_tag = {v:k for k,v in tag_map.items()}
    data = []

    for i,sample in enumerate(conll03_dataset['test']):
        uni_tags = [map_tag('en-ptb', 'universal', int_to_tag[t]) for t in sample['pos_tags']]
        data.append([
            sample['tokens'],
            i,
            [POS_LABEL_MAP[t] for t in uni_tags],
            [POS_LABEL_MAP[t] for t in uni_tags],
            [[0.] * (len(POS_LABEL_MAP) - 1)] * len(uni_tags)
        ])
    dataset = Dataset.from_pandas(
        pd.DataFrame(data, columns=['word', 'id_sent', 'gold', 'majority_label', 'bayes_dist']))
    dataset = dataset.map(partial(preprocess_pos_data, tokenizer), batched=True, remove_columns=dataset.column_names)
    return dataset


def read_rte_transformers(data_dir, tokenizer, aggregation, weights=None, print_accuracy=False, seed=1000, distributions=list(crowdkit_classes.keys())):
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

    if aggregation != 'all_individual':
        class_dist = aggregate(dist_dict,
                               aggregation,
                               weights=weights
        )
        dataset.insert(len(dataset.columns), 'bayes_dist', class_dist.tolist())
        if aggregation == 'label_mixer':
            dataset.insert(len(dataset.columns), 'majority_label', np.argmax(dist_dict['softmax'], -1).tolist())
        else:
            dataset.insert(len(dataset.columns), 'majority_label', np.argmax(class_dist, -1).tolist())
        # dataset['bayes_dist'] = class_dist.tolist()
        # dataset['majority_label'] = np.argmax(class_dist, -1).tolist()
        dataset = Dataset.from_pandas(dataset)
        if print_accuracy:
            pred = np.argmax(class_dist, axis=-1)
            gold = np.array(dataset['value']).astype(np.int32)
            print(f"Accuracy {aggregation}: {(pred==gold).mean()}")

        dataset = dataset.map(partial(preprocess_rte_data, tokenizer), batched=True, remove_columns=dataset.column_names)
        return dataset
    else:
        datasets = {}
        for method in dist_dict:
            d_curr = deepcopy(dataset)
            d_curr['bayes_dist'] = dist_dict[method].tolist()
            d_curr['majority_label'] = np.argmax(dist_dict[method], -1).tolist()
            d_curr = Dataset.from_pandas(d_curr)

            datasets[method] = d_curr.map(partial(preprocess_rte_data, tokenizer), batched=True,
                                  remove_columns=d_curr.column_names)
        return datasets


def read_mre_transformers(data_dir, tokenizer, aggregation, weights=None, print_accuracy=False, seed=1000, distributions=list(crowdkit_classes.keys())):
    mre_raw = pd.read_csv(f"{data_dir}/annotations.csv")
    dataset = pd.read_csv(f"{data_dir}/gold.csv")
    # orig_map = {row['SID']: i for i,row in orig_sents.iterrows()}
    # selection = []
    # for g, group in mre_raw.groupby('orig_id'):
    #     selection.append(orig_map[g])
    #
    # dataset = orig_sents.iloc[selection]
    if os.path.exists(f"{data_dir}/formatted_annotations.csv"):
        annotations_df = pd.read_csv(f"{data_dir}/formatted_annotations.csv")
    else:
        seen = set()
        annotations = {}
        for g, group in mre_raw.groupby('sent_id'):
            annotations[g] = {}
            for j, row in group.iterrows():
                if (g,row['_worker_id']) in seen:
                    annotations[g][row['_worker_id']] = min(1, annotations[g][row['_worker_id']] + int('CAUSE' in row.to_string()))
                else:
                    annotations[g][row['_worker_id']] = int('CAUSE' in row.to_string())
                    seen.add((g,row['_worker_id']))
        # Turn into dframe
        final_annotations = []
        for id_ in annotations:
            for worker in annotations[id_]:
                final_annotations.append([id_, worker, annotations[id_][worker]])
        annotations_df = pd.DataFrame(final_annotations, columns=['task', 'worker', 'label'])
        annotations_df.to_csv(f"{data_dir}/formatted_annotations.csv", index=None)

    annotations, _ = get_annotation_matrix(annotations_df, {i: i for i in range(2)})

    # First get standard and softmax distributions
    dist_dict = {}
    if 'softmax' in distributions:
        dist_dict['softmax'] = softmax_distribution(annotations, {i: i for i in range(2)})
    if 'standard' in distributions:
        dist_dict['standard'] = normalized_distribution(annotations, {i: i for i in range(2)})

    dist_dict = get_crowdkit_distributions(data_dir, dist_dict, annotations_df, distributions=distributions, seed=seed)

    # Now get them into a matrix instead of dict
    for dist in dist_dict:
        dist_dict[dist] = np.array([dist_dict[dist][task] for task in dataset['SID']])

    if aggregation != 'all_individual':

        class_dist = aggregate(dist_dict,
                               aggregation,
                               weights=weights
                               )
        dataset.insert(len(dataset.columns), 'bayes_dist', class_dist.tolist())
        dataset.insert(len(dataset.columns), 'majority_label', np.argmax(class_dist, -1).tolist())
        if print_accuracy:
            pred = np.argmax(class_dist, axis=-1)
            gold = np.array(dataset['gold']).astype(np.int32)
            print(f"Accuracy {aggregation}: {(pred==gold).mean()}")

        dataset = Dataset.from_pandas(dataset)

        dataset = dataset.map(partial(preprocess_mre_data, tokenizer), batched=True,
                              remove_columns=dataset.column_names)
        return dataset
    else:
        datasets = {}
        for method in dist_dict:
            d_curr = deepcopy(dataset)
            d_curr['bayes_dist'] = dist_dict[method].tolist()
            d_curr['majority_label'] = np.argmax(dist_dict[method], -1).tolist()
            d_curr = Dataset.from_pandas(d_curr)

            datasets[method] = d_curr.map(partial(preprocess_mre_data, tokenizer), batched=True,
                                          remove_columns=d_curr.column_names)
        return datasets


def get_original_jigsaw_distribution(data):
    ids = set(data['id'])
    civilcomments_annotations = pd.read_csv('./data/jigsaw/civilcomments_toxicity_individual_annotations.csv')
    civilcomments_annotations = civilcomments_annotations[civilcomments_annotations['id'].isin(ids)]
    annotations = {}
    worker_to_colno = {}
    for i, (g, _) in enumerate(civilcomments_annotations.groupby('worker')):
        worker_to_colno[g] = i

    for g, group in civilcomments_annotations.groupby('id'):
        annotations[g] = [-1] * len(worker_to_colno)
        for j, row in group.iterrows():
            annotations[g][worker_to_colno[row['worker']]] = int((row['toxic'] + row['severe_toxic']) > 0)

    annotations = np.array([annotations[g] for g in data['id']])
    mask = annotations >= 0
    annotations[annotations < 0] = 0

    total_ones = annotations.sum(-1)
    total_zeros = mask.sum(-1) - total_ones
    class_logits = np.vstack([total_zeros, total_ones]).T

    standard_dist = class_logits / class_logits.sum(-1, keepdims=True)
    return standard_dist.tolist()


def read_jigsaw_data_transformers(data_dir, tokenizer, aggregation, weights=None, load_test=False, print_accuracy=False, seed=1000, distributions=list(crowdkit_classes.keys()), in_domain=False):
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

    if aggregation != 'all_individual':

        class_dist = aggregate(dist_dict,
                               aggregation,
                               weights=weights
                               )
        dataset = dataset[['id', 'comment_text', 'gold', 'split', 'target']]
        dataset.insert(len(dataset.columns), 'bayes_dist', class_dist.tolist())
        dataset.insert(len(dataset.columns), 'majority_label', np.argmax(class_dist, -1).tolist())

        if print_accuracy:
            for dist in dist_dict:
                pred = np.argmax(dist_dict[dist], axis=-1)
                gold = np.array(dataset['gold']).astype(np.int32)
                print(f"Accuracy {dist}: {(pred==gold).mean()}")
            pred = np.argmax(class_dist, axis=-1)
            gold = np.array(dataset['gold']).astype(np.int32)
            print(f"Accuracy {aggregation}: {(pred == gold).mean()}")

        train_data = dataset[dataset['split'] == 'train']
        dev_data = dataset[dataset['split'] == 'val']
        test_data = dataset[dataset['split'] == 'test']

        # Get the original jigsaw annotations for the test data
        if load_test:
            test_data['bayes_dist'] = get_original_jigsaw_distribution(test_data)

        if in_domain:
            test_data['gold'] = np.argmax(dist_dict['standard'], -1)[-test_data.shape[0]:]

        dataset = DatasetDict()
        dataset['train'] = Dataset.from_pandas(train_data)
        dataset['validation'] = Dataset.from_pandas(dev_data)
        dataset['test'] = Dataset.from_pandas(test_data)

        dataset = dataset.map(partial(preprocess_jigsaw_data, tokenizer), batched=True,
                              remove_columns=dataset['train'].column_names)

        return dataset

    else:
        datasets = {}
        for method in dist_dict:
            d_curr = deepcopy(dataset)

            d_curr = d_curr[['id', 'comment_text', 'gold', 'split', 'target']]
            d_curr['bayes_dist'] = dist_dict[method].tolist()
            d_curr['majority_label'] = np.argmax(dist_dict[method], -1).tolist()

            train_data = d_curr[d_curr['split'] == 'train']
            dev_data = d_curr[d_curr['split'] == 'val']
            test_data = d_curr[d_curr['split'] == 'test']

            d_curr = DatasetDict()
            d_curr['train'] = Dataset.from_pandas(train_data)
            d_curr['validation'] = Dataset.from_pandas(dev_data)
            d_curr['test'] = Dataset.from_pandas(test_data)

            datasets[method] = d_curr.map(partial(preprocess_jigsaw_data, tokenizer), batched=True,
                                  remove_columns=d_curr['train'].column_names)
        return datasets


class CustomCIFAR10Dataset(torchvision.datasets.CIFAR10):

    def __init__(self, root, transform, majority_label, class_dist, train=True, download=False):
        super().__init__(root, transform=transform, train=train, download=download)
        self.transform = transform
        self.n_images_per_class = 5
        self.n_classes = 10
        self.majority_label = majority_label
        self.class_dist = class_dist

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        im, label = super().__getitem__(index)
        return im, label, self.majority_label[index], self.class_dist[index]


class CustomCINICDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform, majority_label, class_dist, train=True, download=False):
        super().__init__(root, transform=transform)
        self.transform = transform
        self.n_classes = 10
        self.majority_label = majority_label
        self.class_dist = class_dist

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        im, label = super().__getitem__(index)
        return (im,), label, self.majority_label[index], self.class_dist[index]


def read_cifar10h_transformers(data_dir, transforms, aggregation, weights=None, seed=1000, distributions=list(crowdkit_classes.keys())):
    cifar_raw = pd.read_csv(f"{data_dir}/cifar-10h/data/cifar10h-raw.csv").fillna('')
    cifar_raw = cifar_raw[cifar_raw['cifar10_test_test_idx'] >= 0]
    dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar-10h/data/', train=False,
                                             download=True, transform=transforms)

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

    class_dist = aggregate(dist_dict,
                           aggregation,
                           weights=weights
                           )

    dataset.bayes = class_dist
    dataset.majority_label = np.argmax(class_dist, -1)

    dataset = CustomCIFAR10Dataset(root='./data/cifar-10h/', train=False,
                                           download=True, transform=transforms,
                                   majority_label=np.argmax(class_dist, -1), class_dist=class_dist)

    return dataset