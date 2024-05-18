import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoConfig
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from util.datareader import DataCollatorForTokenClassificationBayes
from util.datareader import POS_LABEL_MAP
from util.datareader import read_fornaciarni_pos_transformers, read_penn_treebank_transformers, read_conll03_transformers
from util.datareader import crowdkit_classes
from util.loss import registered_losses
from util.metrics import acc_f1, calculate_nll
from util.model import AutoModelForMultiTaskTokenClassification, LabelMixer
from util.temperature_scaling import ModelWithTemperature
import ipdb

from util.loss import (CrossEntropyLoss,
                       KLLoss,
                       HardSampleLoss)


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


def evaluate(dataloader, model):
    model.eval()
    with torch.no_grad():
        eval_labels = []
        eval_preds = []
        eval_logits = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            label_mask = torch.logical_and(batch['labels'].reshape(-1) < 12, batch['labels'].reshape(-1) >= 0)

            eval_labels.extend(batch['labels'].reshape(-1)[label_mask].cpu().numpy())
            # calculate outputs by running images through the network
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs['logits'].reshape(-1, outputs['logits'].shape[-1])[label_mask, :12]

            eval_logits.extend(logits.cpu().numpy())
            _, predicted = torch.max(logits.data, 1)
            eval_preds.extend(F.softmax(logits, -1).cpu().numpy())

    return eval_labels, eval_logits, eval_preds


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
                        choices=['standard', 'softmax', 'ds', 'mace', 'wawa',
                                 'glad', 'zbs', 'ensemble_basic'], default='ds')
    parser.add_argument("--crowdkit_classes", help="List of crowdkit classes to use for aggregation",
                        required=False, type=str, default=list(crowdkit_classes.keys()) + ['softmax', 'standard'], nargs='+')
    parser.add_argument("--train_majority", help="Whether to use the the majority vote label for training",
                        action="store_true")
    parser.add_argument("--ood_dataset", help="Which ood dataset to use", type=str,
                        choices=['ptb', 'conll03'], default='ptb')
    parser.add_argument("--in_domain", help="If set, will do 3-fold CV using an in-domain test set",
                        action="store_true")

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
    n_labels = len(POS_LABEL_MAP) - 1
    ood_dataset = args.ood_dataset
    distributions = args.crowdkit_classes
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

    dataset = read_fornaciarni_pos_transformers(
        args.data_loc,
        tk,
        aggregation,
        seed=seed,
        distributions=distributions,
        weights=weight_dict
    )

    if args.in_domain:
        testset = dataset['test']
    else:
        if ood_dataset == 'ptb':
            testset = read_penn_treebank_transformers(tk)
        elif ood_dataset == 'conll03':
            testset = read_conll03_transformers(tk)

    collator = DataCollatorForTokenClassificationBayes(tk)
    collator.label_mixer = aggregation == 'label_mixer'
    trainloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size,
                                              num_workers=2, shuffle=True, collate_fn=collator)
    devloader = torch.utils.data.DataLoader(dataset['validation'], batch_size=batch_size,
                                              num_workers=0, shuffle=True, collate_fn=collator)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              num_workers=2, collate_fn=collator)

    # Get the dataset
    config = AutoConfig.from_pretrained(model_name, num_labels=len(POS_LABEL_MAP) - 1)

    if multi_task:
        model = AutoModelForMultiTaskTokenClassification(model_name, config=config)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    model = model.to(device)
    # Save an initial version of the model
    torch.save(model.state_dict(), f'{args.output_dir}/model.pth')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if aggregation == 'label_mixer':
        #mixer = LabelMixer(N=len(dataset['train']), K=len(args.crowdkit_classes)).to(device)
        mixer = LabelMixer(N=1, K=len(args.crowdkit_classes)).to(device)
        optimizer_grouped_parameters.append(
            {'params': mixer.parameters(), 'weight_decay': 0.0}
        )
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        n_epochs * len(trainloader)  # total
    )

    loss_fn = eval(loss_class)().to(device)
    best_F1 = 0.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            batch = {k: v.to(device) for k, v in batch.items()}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            label_mask = torch.logical_and(batch['majority_labels'].reshape(-1) < 12, batch['majority_labels'].reshape(-1) >= 0)
            loss = 0.
            if aggregation == 'label_mixer':
                alpha, kl = mixer(batch['idx'], return_kl=True)
                wandb.log({'kl': kl})
                loss += kl.mean()
                mixed_labels = (batch['bayes'] * alpha.unsqueeze(-1).unsqueeze(-1)).sum(1)
                soft_labels = mixed_labels.reshape(-1, n_labels)[label_mask, :12]
            else:
                soft_labels = batch['bayes'].reshape(-1, n_labels)[label_mask, :12]

            if multi_task:
                loss += loss_fn(
                    soft_labels=soft_labels,
                    labels=batch['majority_labels'].reshape(-1)[label_mask] if train_majority else batch['labels'].reshape(-1)[label_mask],
                    logits=outputs['logits'].reshape(-1, n_labels)[label_mask, :12],
                    multi_task=True,
                    soft_logits=outputs['soft_logits'].reshape(-1, n_labels)[label_mask, :12]
                )
            else:
                loss += loss_fn(
                    soft_labels=soft_labels,
                    labels=batch['majority_labels'].reshape(-1)[label_mask] if train_majority else batch['labels'].reshape(-1)[label_mask],
                    logits=outputs['logits'].reshape(-1, n_labels)[label_mask, :12],
                    multi_task=False
                )
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({
                'loss': loss.item()
            })
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        dev_labels, dev_logits, dev_preds = evaluate(devloader, model)
        acc, P, R, F1 = acc_f1(dev_logits, dev_labels, average='macro')
        if F1 > best_F1:
            print(f"Best validation F1: {F1}")
            best_F1 = F1
            torch.save(model.state_dict(), f'{args.output_dir}/model.pth')
            wandb.log({
                'Val acc': acc,
                'Val P': P,
                'Val R': R,
                'Val F1': F1
            })
    model.load_state_dict(torch.load(f'{args.output_dir}/model.pth'))
    model.eval()
    collator.label_mixer = False
    # First get accuracy metrics
    eval_labels, eval_logits, eval_preds = evaluate(testloader, model)
    acc, P, R, F1 = acc_f1(eval_logits, eval_labels, average='macro')
    log_data = {
        'acc': acc,
        'F1': F1,
        'P': P,
        'R': R
    }


    clls = defaultdict(list)

    indices = list(range(len(testset)))
    for _ in range(5):
        random.shuffle(indices)

        devloader = torch.utils.data.DataLoader(testset.select(indices[:len(indices) // 2]), batch_size=batch_size,
                                                num_workers=2, collate_fn=collator)
        testloader = torch.utils.data.DataLoader(testset.select(indices[len(indices) // 2:]), batch_size=batch_size,
                                                 num_workers=2, collate_fn=collator)

        # Then get scoring and calibration metrics
        temperature_model = ModelWithTemperature(model, device).to(device)
        print("Optimizing temperature on validation data")
        val_metrics = temperature_model.calculate_metrics(devloader, train=True)
        # for k, v in val_metrics.items():
        #     log_data[f'calibrate_{k}'] = v

        print("Calculating calibration metrics on test data")
        test_metrics = temperature_model.calculate_metrics(testloader)
        for k, v in test_metrics.items():
            clls[f'test_{k}'].append(v)
    for metric in clls:
        log_data[metric] = np.mean(clls[metric])
    print(log_data)
    wandb.log(log_data)
    with open(f"{args.metrics_dir}/{seed}.json", 'wt') as f:
        f.write(json.dumps(log_data))

    # Write out test set logits and temperature
    np.save(f"{args.metrics_dir}/logits_{seed}.npy", np.array(eval_logits))

    print('Finished Training')