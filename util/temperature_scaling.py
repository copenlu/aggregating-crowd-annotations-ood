"""
This file comes from this github repository: https://github.com/gpleiss/temperature_scaling

Adapted for use with the models we are using
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
from util.metrics import brier_score
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
import numpy as np
import cvxpy as cp
import jax.numpy as jnp

sns.set()


def temperature_scale(logits, temp):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature


def temperature_scale_distribution(p, temp):
    """
    Perform temperature scaling on probability distribution
    :param p:
    :param temp:
    :return:
    """
    temperature = temp.unsqueeze(1).expand(p.size(0), p.size(1))
    return F.softmax(p.log() / temperature, -1)


class JSD(nn.Module):
    """
    An implementation of the Jensen-Shannon Divergence (needed since it is symmetrical,
    and we wish to minimize the distance between two arbitrary probability distributions)
    """
    def __init__(self):
        super(JSD, self).__init__()
        self.kld = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p, q):
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kld(m, p.log()) + self.kld(m, q.log()))


def calculate_temperature_scaled_ensemble(dist1, dist2, lam=0.001):
    T1 = nn.Parameter(torch.ones(1)*1.5)
    T2 = nn.Parameter(torch.ones(1)*1.5)

    # Calculate in log space
    logits1 = torch.clamp(dist1, min=1e-8, max=1-1e-8).log()
    logits2 = torch.clamp(dist2, min=1e-8, max=1-1e-8).log()
    jsd_criterion = JSD()

    optimizer = optim.LBFGS([T1, T2], lr=0.01, max_iter=1000)

    def temperature_scale(logits, temp):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def eval():
        optimizer.zero_grad()
        dist1 = F.softmax(temperature_scale(logits1, T1), -1)
        dist2 = F.softmax(temperature_scale(logits2, T2), -1)
        loss = jsd_criterion(dist1, dist2) + lam * (T1**2 + T2**2)
        loss.backward()
        return loss

    optimizer.step(eval)

    return 0.5 * (F.softmax(temperature_scale(logits1, T1), -1) + F.softmax(temperature_scale(logits2, T2), -1)), T1, T2

def delta_f(theta):
    norm = np.clip(1 - theta.sum(-1, keepdims=True), a_min=1e-8, a_max=None)
    return np.log(theta / norm)

def delta_f_inv(eta):
    norm = 1 + np.exp(eta).sum(-1, keepdims=True)
    return np.exp(eta) / norm

def calculate_jensen_shannon_centroid(dists, T=1000, weights=None):
    # convert to natural parameters
    natural_dists = [dist[:,:-1] for dist in dists]
    theta = np.average(np.stack(natural_dists), weights=weights, axis=0)
    #theta[:,0] = 1 - theta[:,1:].sum(-1)
    converged = False
    for t in range(T):
        dfs = np.average(np.stack([delta_f(np.stack([theta, dist]).mean(0)) for dist in natural_dists]), weights=weights, axis=0)
        theta_new = delta_f_inv(dfs)
        # Stop if there's no significant difference
        diff = np.abs(theta - theta_new).sum()
        if np.abs(theta - theta_new).sum() < 1e-10:
            print(f"Jensen-Shannon centroid converged after {t} iterations")
            converged = True
            break
        theta = theta_new

    assert diff < 1e-6, f"Couldn't converge after {T} iterations!"
    return np.hstack([theta, 1 - theta.sum(-1,keepdims=True)])


def calculate_temperature_scaled_ensemble_multiple(distances, start=1.5, lam=1e-3, lr=1e-2):
    T = [nn.Parameter(torch.ones(1)*start) for _ in distances]

    # Calculate in log space
    logits = [torch.clamp(dist.log(), min=-15) for dist in distances]

    jsd_criterion = JSD()

    optimizer = optim.LBFGS(T, lr=lr, max_iter=1000)

    def eval():
        optimizer.zero_grad()
        dists = [F.softmax(temperature_scale(logits[i], T[i]), -1) for i in range(len(logits))]
        # dist1 = F.softmax(temperature_scale(logits1, T1), -1)
        # dist2 = F.softmax(temperature_scale(logits2, T2), -1)
        losses = 0.
        reg = 0.0
        n = 0
        for i in range(len(logits)):
            for j in range(i+1, len(logits)):
                losses += jsd_criterion(dists[i], dists[j])
                n += 1
            reg += (lam * T[i]**2)[0]
        loss = (losses / n) + reg
        loss.backward()
        # for t in T:
        #     t.data = t.data.clamp(min=0.5)
        return loss

    optimizer.step(eval)

    return np.mean([F.softmax(temperature_scale(logits[i], T[i]), -1).detach().cpu().numpy() for i in range(len(logits))], 0).tolist(), T
    #return 0.5 * (F.softmax(temperature_scale(logits1, T1), -1) + F.softmax(temperature_scale(logits2, T2), -1)), T1, T2


def calculate_temperature_scaled_ensemble_maximize(distances, start=1.5, lam=1e-3, lr=1e-2):
    T = [nn.Parameter(torch.ones(1)*start) for _ in distances]

    # Calculate in log space
    logits = [torch.clamp(dist.log(), min=-15) for dist in distances]

    jsd_criterion = JSD()

    optimizer = optim.SGD(T, lr=lr, maximize=True, momentum=0.9)

    loss_curr = 0.
    for i in range(1000):
        optimizer.zero_grad()
        dists = [F.softmax(temperature_scale(logits[i], T[i]), -1) for i in range(len(logits))]
        # dist1 = F.softmax(temperature_scale(logits1, T1), -1)
        # dist2 = F.softmax(temperature_scale(logits2, T2), -1)
        losses = 0.
        reg = 0.0
        n = 0
        for i in range(len(logits)):
            for j in range(i+1, len(logits)):
                losses += jsd_criterion(dists[i], dists[j])
                n += 1
            reg += (lam * T[i]**2)[0]
        loss = (losses / n) + reg
        loss.backward()
        optimizer.step()
        # for t in T:
        #     t.data = t.data.clamp(min=0.5)
        # if abs(loss.item() - loss_curr) < 1e-5:
        #     break
        loss_curr = loss.item()

    return np.mean([F.softmax(temperature_scale(logits[i], T[i]), -1).detach().cpu().numpy() for i in range(len(logits))], 0).tolist(), T
    #return 0.5 * (F.softmax(temperature_scale(logits1, T1), -1) + F.softmax(temperature_scale(logits2, T2), -1)), T1, T2



def calculate_temperature_scaled_ensemble_centroid(distances, start=1.5, lam=1e-3, lr=10):
    # TODO: make T for each item
    #T = nn.Parameter(torch.ones(1)*start) #[nn.Parameter(torch.ones(1)*start) for _ in distances]

    # Calculate in log space
    logits = [torch.clamp(dist.log(), min=-15)[0] for dist in distances]
    Q = nn.Parameter(torch.stack([F.softmax(log, -1) for log in logits]).mean(0))
    #Q = nn.Parameter(torch.ones(2))
    ipdb.set_trace()

    jsd_criterion = JSD()
    xent = nn.CrossEntropyLoss()

    optimizer = optim.LBFGS([Q], lr=lr, max_iter=1000)

    def temperature_scale(logits, temp):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def eval():
        optimizer.zero_grad()
        ipdb.set_trace()
        dists = [F.softmax(log, -1) for log in logits]
        #Q_curr = F.softmax(Q, -1)
        # dist1 = F.softmax(temperature_scale(logits1, T1), -1)
        # dist2 = F.softmax(temperature_scale(logits2, T2), -1)
        losses = 0.
        reg = 0.0
        n = 0
        for dist in dists:
            losses += jsd_criterion(Q, dist)
            #losses += xent(Q.unsqueeze(0), dist.unsqueeze(0))
            n += 1
            #reg += (lam * T[i]**2)[0]
        loss = (losses / n)
        loss.backward()
        # for t in T:
        #     t.data = t.data.clamp(min=0.5)
        return loss

    optimizer.step(eval)

    return F.softmax(Q, -1).detach().cpu().numpy().tolist()
    #return 0.5 * (F.softmax(temperature_scale(logits1, T1), -1) + F.softmax(temperature_scale(logits2, T2), -1)), T1, T2



class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return self.temperature_scale(outputs['logits'])

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


    def get_logits_and_labels(self, valid_loader):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs['logits']
                labels = batch['labels']
                if len(labels.shape) == 2:
                    # Hard coding label ranges, only have one token level task
                    label_mask = torch.logical_and(batch['labels'].reshape(-1) < 12, batch['labels'].reshape(-1) >= 0)
                    logits = logits.reshape(-1, logits.shape[-1])[label_mask]
                    labels = labels.reshape(-1)[label_mask]

                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return logits, labels


    # This function probably should live outside of this class, but whatever
    def calculate_metrics(self, valid_loader, train=False):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        print(self.temperature)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits, labels = self.get_logits_and_labels(valid_loader)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        before_temperature_brier = brier_score(logits, labels).item()
        #print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        if train:
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.temperature_scale(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_brier = brier_score(self.temperature_scale(logits), labels).item()
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        metrics = {
            'NLL_pre': before_temperature_nll,
            'ECE_pre': before_temperature_ece,
            'Brier_pre': before_temperature_brier,
            'NLL_post': after_temperature_nll,
            'ECE_post': after_temperature_ece,
            'Brier_post': after_temperature_brier
        }
        return metrics


class ViTModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ViTModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return self.temperature_scale(outputs['logits'])

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


    def get_logits_and_labels(self, valid_loader):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for ((inputs,), labels, majority_labels, class_dist) in valid_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                majority_labels = majority_labels.to(self.device)
                class_dist = class_dist.to(self.device)
                outputs = self.model(inputs)
                logits = outputs['logits']

                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return logits, labels


    # This function probably should live outside of this class, but whatever
    def calculate_metrics(self, valid_loader, train=False):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        print(self.temperature)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits, labels = self.get_logits_and_labels(valid_loader)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        before_temperature_brier = brier_score(logits, labels).item()
        #print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        if train:
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.temperature_scale(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_brier = brier_score(self.temperature_scale(logits), labels).item()
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        metrics = {
            'NLL_pre': before_temperature_nll,
            'ECE_pre': before_temperature_ece,
            'Brier_pre': before_temperature_brier,
            'NLL_post': after_temperature_nll,
            'ECE_post': after_temperature_ece,
            'Brier_post': after_temperature_brier
        }
        return metrics



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
