import torch
from torch import nn
from torch.optim import SGD
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple
import ipdb


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return np.sum(np.argmax(logits, axis=-1) == labels).astype(np.float32) / float(labels.shape[0])


def acc_f1(logits: List, labels: List, average='macro') -> Tuple[float, float, float, float]:
    labels = np.asarray(labels).reshape(-1)
    if type(logits[0]) == list:
        # Ensemble N x K x L
        all_logits = np.array(logits)
        all_preds = np.argmax(all_logits, -1)
        preds = np.argmax(np.bincount(all_preds), -1)
        acc = (preds == labels).mean()
    else:
        logits = np.asarray(logits).reshape(-1, len(logits[0]))
        acc = accuracy(logits, labels)
    P, R, F1, _ = precision_recall_fscore_support(labels, np.argmax(logits, axis=-1), average=average)
    return acc,P,R,F1


def brier_score(logits, labels):
    preds = F.softmax(logits, dim=-1)
    return 1 + (torch.sum(preds ** 2) - 2 * torch.sum(preds[torch.arange(preds.shape[0]), labels])) / labels.shape[0]


def expected_calibration_error(y_true, y_pred, num_bins=15):
    """From https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
    """
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]


def calculate_nll(labels, logits):
    labels = torch.tensor(labels)
    logits = torch.tensor(logits)
    preds = F.log_softmax(logits, dim=-1)
    if len(labels.shape) == 2:
        label_mask = torch.logical_and(labels.reshape(-1) < 12, labels.reshape(-1) >= 0)
        preds = preds.reshape(-1, preds.shape[-1])[label_mask]
        labels = labels.reshape(-1)[label_mask]
    nll = F.nll_loss(preds, labels, reduction='none')
    return float(nll.detach().cpu().numpy().mean())

def negative_log_likelihood(model, testloader, device, T=None):
    with torch.no_grad():
        likelihoods = []
        # Measure the negative log likelihood on the held out set
        for i, batch in enumerate(tqdm(testloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            batch = {k: v.to(device) for k,v in batch.items()}

            # forward + backward + optimize
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs['logits']
            if T:
                logits = T(logits)
            preds = F.log_softmax(logits, dim=-1)
            if len(batch['labels'].shape) == 2:
                label_mask = torch.logical_and(batch['labels'].reshape(-1) < 12, batch['labels'].reshape(-1) >= 0)
                preds = preds.reshape(-1, preds.shape[-1])[label_mask]
                batch['labels'] = batch['labels'].reshape(-1)[label_mask]
            nll = F.nll_loss(preds, batch['labels'], reduction='none')
            likelihoods.append(nll.detach().cpu().numpy())
    return np.concatenate(likelihoods).mean()


def calibrated_log_likelihood(model, testset, device, collate_fn, n_runs=5):
    neg_log_likelihood = 0.0
    # Run n times
    for i in range(n_runs):
        # Divide the test set into 2
        train_idx, valid_idx = train_test_split(
            np.arange(len(testset['labels'])),
            test_size=0.5,
            shuffle=True,
            stratify=testset['labels'] if type(testset['labels']) == int else None)
        calibration_sampler = torch.utils.data.SubsetRandomSampler(train_idx.tolist())
        test_sampler = torch.utils.data.SubsetRandomSampler(valid_idx.tolist())
        calibration_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                  num_workers=2, sampler=calibration_sampler,
                                                  collate_fn=collate_fn)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                  num_workers=2, sampler=test_sampler,
                                                  collate_fn=collate_fn)

        # Optimize the softmax temperature to minimize the negative log likelihood
        class Temp(nn.Module):
            def __init__(self):
                super().__init__()
                self.T = nn.Parameter(torch.ones(1))
            def forward(self, logits):
                return logits / self.T
        T = Temp().to(device)
        optim = SGD(T.parameters(), lr=1e-3)
        patience = 10
        c = 0
        eps = 1e-4
        t_curr = 1.0
        nll_curr = float('inf')
        done = False
        #for epoch in range(10):  # loop over the dataset multiple times
        for i, batch in enumerate(tqdm(calibration_loader), 0):
            # get the inputs; data is a list of [inputs, labels]
            batch = {k: v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs['logits']
            logits = T(logits)
            preds = F.log_softmax(logits, dim=-1)
            if len(batch['labels'].shape) == 2:
                label_mask = torch.logical_and(batch['labels'].reshape(-1) < 12, batch['labels'].reshape(-1) >= 0)
                preds = preds.reshape(-1, preds.shape[-1])[label_mask]
                batch['labels'] = batch['labels'].reshape(-1)[label_mask]
            nll = F.nll_loss(preds, batch['labels'], reduction='mean')
            nll.backward()
            optim.step()
            if abs(t_curr - T.T.item()) > eps:
                c = 0
            else:
                c += 1
                if c == patience:
                    done = True
                    break
            t_curr = T.T.item()
        # if done:
        #     break
        neg_log_likelihood += negative_log_likelihood(model, testloader, device, T)

    return neg_log_likelihood / n_runs