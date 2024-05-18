from torch import nn
import torch
from torch.distributions import Dirichlet
import torch.nn.functional as F
import ipdb


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits, labels, soft_labels=None, **kwargs):
        if len(logits.shape) > 2:
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
        if soft_labels is not None:
            labels = soft_labels
        if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
            return F.cross_entropy(logits, labels, weight=kwargs['class_weights'])
        else:
            return F.cross_entropy(logits, labels)


class KLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits, soft_labels, multi_task, **kwargs):

        if multi_task:
            preds = F.log_softmax(kwargs['soft_logits'], -1)
            # Sample
            #sampled_targets = torch.log(torch.clamp(soft_labels, min=1e-6, max=1-1e-6))
            # Get the KL-divergence
            if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
                kl = F.kl_div(preds, soft_labels, reduction='none')
                hard_labs = torch.max(soft_labels, -1)[1]
                weights = kwargs['class_weights'][hard_labs]
                kld = (weights.unsqueeze(1) * kl).sum(-1).mean()
                ce = F.cross_entropy(logits, kwargs['labels'], weight=kwargs['class_weights'])
            else:
                kld = F.kl_div(preds, soft_labels, reduction='batchmean')
                ce = F.cross_entropy(logits, kwargs['labels'])

            return ce + kld
        else:
            preds = F.log_softmax(logits, -1)
            # Sample
            #sampled_targets = torch.log(torch.clamp(soft_labels, min=1e-6, max=1-1e-6))
            # Get the KL-divergence
            if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
                kl = F.kl_div(preds, soft_labels, reduction='none')
                hard_labs = torch.max(soft_labels, -1)[1]
                weights = kwargs['class_weights'][hard_labs]
                return (weights.unsqueeze(1) * kl).sum() / logits.shape[0]
            else:
                return F.kl_div(preds, soft_labels, reduction='batchmean')


class MixtureLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits, soft_labels, alpha, **kwargs):

        preds = [F.log_softmax(l, -1) for l in logits]
        # Sample
        #sampled_targets = torch.log(torch.clamp(soft_labels, min=1e-6, max=1-1e-6))
        # Get the KL-divergence
        if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
            kl = [F.kl_div(p, labs, reduction='none') for p,labs in zip(preds, soft_labels)]
            hard_labs = torch.max(soft_labels[0], -1)[1]
            weights = kwargs['class_weights'][hard_labs]
            return torch.cat([a*(weights.unsqueeze(1) * k).sum() / l.shape[0] for a,k,l in zip(alpha, kl, logits)]).mean()
        else:
            return torch.cat([a*F.kl_div(p, l, reduction='batchmean') for a,p,l in zip(alpha, preds, soft_labels)]).mean()


class HardSampleLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, logits, soft_labels, **kwargs):
        # Sample
        sampled_targets = torch.multinomial(soft_labels, 1).squeeze(-1)
        # Measure cross entropy between these two
        if 'class_weights' in kwargs and kwargs['class_weights'] is not None:
            return F.cross_entropy(logits, sampled_targets, weight=kwargs['class_weights'])
        else:
            return F.cross_entropy(logits, sampled_targets)


registered_losses = [
    'CrossEntropyLoss',
    'KLLoss',
    'HardSampleLoss',
    'MixtureLoss'
]