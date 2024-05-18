import argparse
import os
import json
import random
from functools import partial
from collections import defaultdict
from pathlib import Path
import ipdb

import numpy as np
import torch
import wandb
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from util.datareader import read_rte_transformers, read_mre_transformers
from util.datareader import crowdkit_classes
from util.eval import run_test_metrics
from util.loss import registered_losses
from util.model import ModelEnsembleAverage, LabelMixer
from util.train import train_cls


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def preprocess_snli_data(tk, examples):
    batch = tk(examples['premise'], text_pair=examples['hypothesis'])
    # Map these to the RTE data
    label_map = {0: 1, 1: 0, 2: 0}
    batch['labels'] = [label_map[v] for v in examples['label']]
    batch['majority_labels'] = batch['labels']
    # Unused, doesn't matter
    batch['bayes'] = [[0.,0.]]*len(examples['label'])

    return batch


def preprocess_exaggeration_data(tk, examples):
    batch = tk(examples['sentence'])
    # Map these to the RTE data
    label_map = {0: 0, 1: 0, 2: 0, 3: 1}
    batch['labels'] = [label_map[v] for v in examples['label']]
    batch['majority_labels'] = batch['labels']
    # Unused, doesn't matter
    batch['bayes'] = [[0., 0.]] * len(examples['label'])

    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc",
                        help="The location of the dataset",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='roberta-base')
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--n_gpu", help="The number of gpus to use", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--metrics_dir", help="Directory to store metrics for making latex tables", required=True,
                        type=str)
    parser.add_argument("--loss_function", help="The class name of the loss function to use", type=str,
                        choices=registered_losses, default='CrossEntropyLoss')
    parser.add_argument("--multi_task", help="Whether to use multi-task learning", action="store_true")
    parser.add_argument("--aggregation", help="Which label aggregation method to use", type=str,
                        choices=['standard', 'softmax', 'ds', 'mace', 'glad', 'wawa',
                                 'zbs', 'ensemble_basic'], default='ds')
    parser.add_argument("--crowdkit_classes", help="List of crowdkit classes to use for aggregation",
                        required=False, type=str, default=list(crowdkit_classes.keys()) + ['softmax', 'standard'], nargs='+')
    parser.add_argument("--train_majority", help="Whether to use the the majority vote label for training",
                        action="store_true")
    parser.add_argument("--use_class_weights", help="Whether to weight the loss based on class imbalances (for MRE)",
                        action="store_true")
    parser.add_argument("--in_domain", help="If set, will do 3-fold CV using an in-domain test set",
                        action="store_true")
    parser.add_argument("--task", help="Which task to run", type=str,
                        choices=['rte', 'mre'], default='rte')
    parser.add_argument("--ood_dataset", help="Which ood dataset to use", type=str,
                        choices=['snli', 'anli'], default='snli')



    args = parser.parse_args()
    seed = args.seed
    model_name = args.model_name
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    loss_class = args.loss_function
    multi_task = args.multi_task
    aggregation = args.aggregation
    train_majority = args.train_majority
    n_splits = 3
    task = args.task
    ood_dataset = args.ood_dataset
    use_class_weights = args.use_class_weights
    distributions = args.crowdkit_classes
    class_weights = None
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")
    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")

    if multi_task:
        assert args.loss_function == 'KLLoss', "Multi-task learning only compatible with KL Loss"
    if train_majority:
        assert args.loss_function == 'CrossEntropyLoss', "Majority voting should only be experimented with xentropy"
        assert aggregation in ['softmax', 'standard'], "Majority voting should be used with baseline labels"

    # Enforce reproducibility
    # Always first
    enforce_reproducibility(seed)
    config = {
        'run_name': args.run_name,
        'seed': seed,
        'model_name': model_name,
        'output_dir': args.output_dir,
        'tags': args.tags,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'epochs': args.n_epochs,
        'seed': args.seed,
        'loss_fn': loss_class,
        'multi_task': multi_task,
        'aggregation': aggregation,
        'train_majority': train_majority,
        'task': task,
        'ood_dataset': ood_dataset,
        'crowdkit_classes': distributions,
        'in_domain': args.in_domain
    }

    run = wandb.init(
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tags
    )

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    tk = AutoTokenizer.from_pretrained(model_name)

    weight_dict = None
    if aggregation == 'ensemble_nll':
        # Get the weights based on dev NLL
        metric_dir_parent = Path(args.metrics_dir).parent
        weight_dict = {}
        nll_sum = 0.
        for method in list(crowdkit_classes.keys()) + ['softmax', 'standard']:
            with open(metric_dir_parent / f'kld-{method}/{args.seed}.json') as f:
                data = json.load(f)
                weight_dict[method] = data['dev_nll']
                nll_sum += data['dev_nll']
        # Get the min value and percentage of min value
        min_nll = min(weight_dict.values())
        for method in weight_dict:
            weight_dict[method] = min_nll / weight_dict[method]

        total = sum(weight_dict.values())
        for method in weight_dict:
            weight_dict[method] = weight_dict[method] / total
        aggregation = 'ensemble_basic'

    if task == 'rte':
        dataset = read_rte_transformers(
            args.data_loc,
            tk,
            aggregation,
            seed=seed,
            distributions=distributions,
            weights=weight_dict
        )

        if ood_dataset == 'snli':
            test_dataset = load_dataset(ood_dataset)['validation'].filter(lambda example: example['label'] >= 0)
        elif ood_dataset == 'anli':
            test_dataset = load_dataset(ood_dataset)['dev_r1'].filter(lambda example: example['label'] >= 0)

        testset = test_dataset.map(partial(preprocess_snli_data, tk), batched=True, remove_columns=test_dataset.column_names)
    elif task == 'mre':
        dataset = read_mre_transformers(
            args.data_loc,
            tk,
            aggregation,
            seed=seed,
            distributions=distributions,
            weights=weight_dict
        )

        if use_class_weights:
            if loss_class == 'KLLoss' or loss_class == 'HardSampleLoss':
                # Get averages for KL divergence
                labels = np.argmax(np.array(dataset['bayes']), -1)
            else:
                labels = np.array(dataset['labels'])
            weight = torch.tensor(len(labels) / (2 * np.bincount(labels)))
            class_weights = weight.type(torch.FloatTensor).to(device)
        test_dataset = load_dataset('copenlu/scientific-exaggeration-detection')
        test_dataset = Dataset.from_dict({
            'sentence': test_dataset['test']['press_release_conclusion'] + test_dataset['test']['abstract_conclusion'],
            'label': test_dataset['test']['press_release_strength'] + test_dataset['test']['abstract_strength']
        })
        testset = test_dataset.map(partial(preprocess_exaggeration_data, tk), batched=True,
                                   remove_columns=test_dataset.column_names)

    collator = DataCollatorWithPadding(tk)


    if args.in_domain:
        # models = []

        metrics_list = defaultdict(list)
        for _ in range(n_splits):
            train_indices = list(range(len(dataset)))
            # Get train and dev
            train_indices, dev_indices = train_test_split(train_indices, test_size=0.2, stratify=dataset['labels'])

            test_indices = dev_indices[len(dev_indices)//2:]
            dev_indices = dev_indices[:len(dev_indices)//2]

            trainset = Dataset.from_dict(dataset[train_indices])
            devset = Dataset.from_dict(dataset[dev_indices])
            testset = Dataset.from_dict(dataset[test_indices])

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      num_workers=2, shuffle=True, collate_fn=collator)
            devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size,
                                                    num_workers=2, shuffle=True, collate_fn=collator)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     num_workers=2, collate_fn=collator)

            # models.append(
            model = train_cls(
                wandb,
                model_name,
                trainloader,
                devloader,
                loss_class,
                multi_task,
                device,
                args.output_dir,
                weight_decay,
                args.warmup_steps,
                n_epochs,
                train_majority,
                class_weights,
                lr=args.learning_rate
            )
            # ))


            curr_metrics = run_test_metrics(
                args.metrics_dir,
                seed,
                wandb,
                testset,
                testloader,
                devloader,
                # ModelEnsembleAverage(models).to(device),
                model.to(device),
                device,
                collator,
                batch_size,
                log_metrics=False,
                return_metrics=True,
                seed=seed
            )
            for m in curr_metrics:
                metrics_list[m].append(curr_metrics[m])
        out_metrics = {}
        for m in metrics_list:
            out_metrics[m] = np.mean(metrics_list[m])

        wandb.log(out_metrics)
        print(out_metrics)
        with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
            f.write(json.dumps(out_metrics))


    else:

        #models = []

        #for _ in range(n_splits):
        train_indices = list(range(len(dataset)))
        # Get train and dev
        train_indices, dev_indices = train_test_split(train_indices, test_size=0.1, stratify=dataset['labels'])

        trainset = Dataset.from_dict(dataset[train_indices])
        devset = Dataset.from_dict(dataset[dev_indices])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  num_workers=2, shuffle=True, collate_fn=collator)
        devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size,
                                                  num_workers=2, shuffle=True, collate_fn=collator)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  num_workers=2, collate_fn=collator)



        #models.append(
        model = train_cls(
            wandb,
            model_name,
            trainloader,
            devloader,
            loss_class,
            multi_task,
            device,
            args.output_dir,
            weight_decay,
            args.warmup_steps,
            n_epochs,
            train_majority,
            class_weights,
            lr=args.learning_rate
        )
        #))

        run_test_metrics(
            args.metrics_dir,
            seed,
            wandb,
            testset,
            testloader,
            devloader,
            #ModelEnsembleAverage(models).to(device),
            model.to(device),
            device,
            collator,
            batch_size,
            seed=seed
        )