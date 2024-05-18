import os
import json
import argparse
import torch
import random
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import ViTConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForTokenClassification
from util.metrics import (expected_calibration_error,
                           calibrated_log_likelihood,
                           negative_log_likelihood)
from util.loss import registered_losses
from util.loss import (CrossEntropyLoss,
                       KLLoss,
                       HardSampleLoss)
from util.model import ViTForMultiTaskImageClassification
from util.metrics import acc_f1, calculate_nll
from util.temperature_scaling import ViTModelWithTemperature

from util.datareader import read_cifar10h_transformers
from util.datareader import CustomCIFAR10Dataset, CustomCINICDataset
from util.datareader import crowdkit_classes
from transformers import DataCollatorWithPadding
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from sklearn.model_selection import train_test_split

import ipdb


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
        for ((inputs,), labels, majority_labels, class_dist) in tqdm(dataloader):
            inputs = inputs.to(device)
            eval_labels.extend(list(labels.detach().cpu().numpy()))
            outputs = model(inputs)
            logits = outputs['logits']
            eval_logits.extend(logits.detach().cpu().numpy())
            _, predicted = torch.max(logits.data, 1)
            eval_preds.extend(list(F.softmax(logits, -1).detach().cpu().numpy()))
            # total += batch['labels'].size(0)
            # correct += (predicted == batch['labels']).sum().item()
    return eval_labels, eval_logits, eval_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_loc",
                        help="The location of the dataset",
                        type=str, required=True)
    parser.add_argument("--model_name",
                        help="The name of the model to train. Can be a directory for a local model",
                        type=str, default='google/vit-base-patch16-224-in21k')
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
                                 'zbs', 'ensemble_basic'],
                        default='ds')
    parser.add_argument("--crowdkit_classes", help="List of crowdkit classes to use for aggregation",
                        required=False, type=str, default=list(crowdkit_classes.keys()) + ['softmax', 'standard'],
                        nargs='+')
    parser.add_argument("--train_majority", help="Whether to use the the majority vote label for training",
                        action="store_true")
    parser.add_argument("--use_class_weights", help="Whether to weight the loss based on class imbalances (for MRE)",
                        action="store_true")
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
    use_class_weights = args.use_class_weights
    class_weights = None
    crowdkit_classes = args.crowdkit_classes
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
        'crowdkit_classes': crowdkit_classes,
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

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _ugmmvae_transforms = Compose(
        [ToTensor(),
         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )


    def train_transforms(image):
        return [_train_transforms(image.convert("RGB"))]

    def val_transforms(image):
        return _val_transforms(image.convert("RGB"))


    classes = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    id2label = {i: l for i, l in enumerate(classes)}
    label2id = {l: i for i, l in id2label.items()}

    print("Load the train data")

    trainset = read_cifar10h_transformers(
        args.data_loc,
        train_transforms,
        aggregation,
        seed=seed,
        distributions=crowdkit_classes
    )

    if args.in_domain:
        train_idx, valtest_idx = train_test_split(
            np.arange(len(trainset.targets)),
            test_size=0.2,
            shuffle=True,
            stratify=trainset.targets,
            random_state=10
        )
        valid_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=0.5,
            shuffle=True,
            stratify=np.array(trainset.targets)[valtest_idx],
            random_state=10
        )
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  num_workers=2, sampler=train_sampler)

        devloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                num_workers=2, sampler=valid_sampler)
        testset = torch.utils.data.Subset(trainset, test_idx)
        testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                num_workers=2, sampler=test_sampler)

    else:
        train_idx, valid_idx = train_test_split(
            np.arange(len(trainset.targets)),
            test_size=0.1,
            shuffle=True,
            stratify=trainset.targets,
            random_state=10
        )
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  num_workers=2, sampler=train_sampler)

        devloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  num_workers=2, sampler=valid_sampler)

        print("Load the test data")

        testset = CustomCINICDataset(root='./data/cifar10h/cinic10/imagenet/', train=True,
                                               download=True, transform=val_transforms,
                                       majority_label=[0]*210000, class_dist=[0]*210000)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

    if multi_task:
        model = ViTForMultiTaskImageClassification(model_name,
                                                   ViTConfig.from_pretrained(model_name),
                                                  num_labels=10,
                                                  id2label=id2label,
                                                  label2id=label2id).to(device)
    else:
        model = ViTForImageClassification.from_pretrained(model_name,
                                                          num_labels=10,
                                                          id2label=id2label,
                                                          label2id=label2id).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        n_epochs * len(trainloader)  # total
    )

    loss_fn = eval(loss_class)().to(device)
    best_F1 = -1.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, ((inputs,), labels, majority_labels, class_dist) in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            majority_labels = majority_labels.to(device)
            class_dist = class_dist.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # TODO: need a multitask version of the model
            if multi_task:
                loss = loss_fn(
                    soft_labels=class_dist,
                    labels=majority_labels if train_majority else labels,
                    logits=outputs['logits'],
                    multi_task=True,
                    soft_logits=outputs['soft_logits'],
                    class_weights=class_weights
                )
            else:
                loss = loss_fn(
                    soft_labels=class_dist,
                    labels=majority_labels if train_majority else labels,
                    logits=outputs['logits'],
                    multi_task=False,
                    class_weights=class_weights
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
    # First get accuracy metrics
    eval_labels, eval_logits, eval_preds = evaluate(testloader, model)
    # Flip the labels so the evaluation is wrt the negative class
    acc, P, R, F1 = acc_f1(eval_logits, eval_labels, average='macro')
    log_data = {
        'acc': acc,
        'F1': F1,
        'P': P,
        'R': R
    }
    dev_labels, dev_logits, dev_preds = evaluate(devloader, model)
    # Flip the labels so the evaluation is wrt the negative class
    d_acc, dP, dR, dF1 = acc_f1(dev_logits, dev_labels, average='macro')
    d_nll = calculate_nll(np.array(dev_labels), np.array(dev_logits))
    log_data.update({
        'dev_acc': d_acc,
        'dev_P': dP,
        'dev_R': dR,
        'dev_F1': dF1,
        'dev_nll': d_nll
    })

    clls = defaultdict(list)
    indices = list(range(len(testset)))
    for _ in range(3):
        random.shuffle(indices)

        devloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, indices[:len(indices) // 2]), batch_size=batch_size,
                                                num_workers=2)
        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, indices[len(indices) // 2:]), batch_size=batch_size,
                                                 num_workers=2)

        # Then get scoring and calibration metrics
        temperature_model = ViTModelWithTemperature(model, device).to(device)
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
    # Save the temperature
    # with open(f"{args.metrics_dir}/temperature_{seed}.txt", 'wt') as f:
    #     f.write(f"{temperature_model.temperature.item()}\n")

    print('Finished Training')