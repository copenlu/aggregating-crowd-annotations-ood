import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet
from torch.special import gammaln, digamma
from transformers import AutoModel
from transformers import ViTModel
import torch.nn.functional as F


class ModelEnsembleAverage(nn.Module):
    def __init__(self, models):
        super(ModelEnsembleAverage, self).__init__()

        self.models = nn.ModuleList(models)

    def forward(self, input_ids, attention_mask):
        logits = []

        for model in self.models:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits.append(outputs['logits'].unsqueeze(0))
        outputs['logits'] = torch.mean(torch.vstack(logits), axis=0)

        return outputs


class AutoModelForMultiTaskSequenceClassification(nn.Module):
    def __init__(self, model_name, config):
        super(AutoModelForMultiTaskSequenceClassification, self).__init__()
        self.config = config
        self.lm = AutoModel.from_pretrained(model_name, config=config)

        # Create the classifiers
        self.cls_hard = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_soft = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, input_ids, attention_mask, **kwargs):
        # Pass through bert
        model_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooler_output = model_outputs['pooler_output']

        model_outputs['logits'] = self.cls_hard(pooler_output)
        model_outputs['soft_logits'] = self.cls_soft(pooler_output)

        return model_outputs


class AutoModelForMixtureSequenceClassification(nn.Module):
    def __init__(self, model_name, config, K=2):
        super(AutoModelForMultiTaskSequenceClassification, self).__init__()
        self.config = config
        self.lm = AutoModel.from_pretrained(model_name, config=config)

        # Mixture parameters
        self.alpha = nn.Parameter(torch.zeros(K), requires_grad=True)

        # Create the classifiers
        self.cls = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels)]*k)


    def forward(self, input_ids, attention_mask, **kwargs):
        # Pass through bert
        model_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooler_output = model_outputs['pooler_output']

        model_outputs['logits'] = [cls(pooler_output) for cls in self.cls]
        model_outputs['alpha'] = F.softmax(self.alpha)
        #model_outputs['soft_logits'] = self.cls_soft(pooler_output)

        return model_outputs


class AutoModelForMultiTaskTokenClassification(nn.Module):
    def __init__(self, model_name, config):
        super(AutoModelForMultiTaskTokenClassification, self).__init__()
        self.config = config
        self.lm = AutoModel.from_pretrained(model_name, config=config)

        # Create the classifiers
        self.cls_hard = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_soft = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, input_ids, attention_mask, **kwargs):
        # Pass through bert
        model_outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooler_output = model_outputs['last_hidden_state'].reshape(-1, self.config.hidden_size)

        model_outputs['logits'] = self.cls_hard(pooler_output)
        model_outputs['soft_logits'] = self.cls_soft(pooler_output)

        return model_outputs


class ViTForMultiTaskImageClassification(nn.Module):
    def __init__(self, model_name, config, num_labels, id2label, label2id):
        super(ViTForMultiTaskImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name,
                                            num_labels=num_labels,
                                            id2label=id2label,
                                            label2id=label2id,
                                            add_pooling_layer=False)

        # Create the classifiers
        self.cls_hard = nn.Linear(config.hidden_size, num_labels)
        self.cls_soft = nn.Linear(config.hidden_size, num_labels)


    def forward(self, pixel_values, **kwargs):
        # Pass through bert
        outputs = self.vit(pixel_values, return_dict=True)
        sequence_output = outputs[0]

        outputs['logits'] = self.cls_hard(sequence_output[:, 0, :])
        outputs['soft_logits'] = self.cls_soft(sequence_output[:, 0, :])

        return outputs


class LabelMixer(nn.Module):
    def __init__(self, N, K):
        super(LabelMixer, self).__init__()

        self.N = N
        self.K = K

        self.logc = nn.Embedding(N,K)
        self.logc.weight.data.zero_()

        self.prior_a = nn.Parameter(torch.ones(1,K), requires_grad=False)

    def forward(self, idx, return_kl=True):
        if self.N > 1:
            c = self.logc(idx).exp()
        else:
            c = self.logc.weight.exp()
        alpha = Dirichlet(c).rsample()
        if return_kl:
            kl = self._kl(c)
            return alpha, kl
        else:
            return alpha

    def _kl(self, alpha):
        # https://statproofbook.github.io/P/dir-kl.html
        qsum = alpha.sum(-1, keepdim=True)
        psum = self.prior_a.sum(-1, keepdim=True)

        term1 = gammaln(qsum) - gammaln(psum)

        term2 = (gammaln(self.prior_a) - gammaln(alpha)).sum(-1, keepdim=True)

        term3 = ((alpha - self.prior_a) * (digamma(alpha) - digamma(qsum))).sum(-1, keepdim=True)

        return term1 + term2 + term3