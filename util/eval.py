import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from util.metrics import acc_f1, calculate_nll
from util.temperature_scaling import ModelWithTemperature


def evaluate(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        eval_labels = []
        eval_preds = []
        eval_logits = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            eval_labels.extend(list(batch['labels'].detach().cpu().numpy()))
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs['logits']
            eval_logits.extend(logits.detach().cpu().numpy())
            _, predicted = torch.max(logits.data, 1)
            eval_preds.extend(list(F.softmax(logits, -1).detach().cpu().numpy()))

    return eval_labels, eval_logits, eval_preds


def run_test_metrics(
        metrics_dir,
        filename,
        logger,
        testset,
        testloader,
        orig_devloader,
        model,
        device,
        collator,
        batch_size = 16,
        return_metrics = False,
        log_metrics = True,
        seed = 1000
):
    model.eval()
    # First get accuracy metrics
    eval_labels, eval_logits, eval_preds = evaluate(testloader, model, device)
    # Flip the labels so the evaluation is wrt the negative class
    acc, P, R, F1 = acc_f1(eval_logits, eval_labels, average='macro')
    log_data = {
        'acc': acc,
        'F1': F1,
        'P': P,
        'R': R
    }

    dev_labels, dev_logits, dev_preds = evaluate(orig_devloader, model, device)
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
    temps = []
    for _ in range(5):
        random.shuffle(indices)

        devloader = torch.utils.data.DataLoader(testset.select(indices[:len(indices) // 2]), batch_size=batch_size,
                                                num_workers=2, collate_fn=collator)
        testloader = torch.utils.data.DataLoader(testset.select(indices[len(indices) // 2:]), batch_size=batch_size,
                                                 num_workers=2, collate_fn=collator)

        # Then get scoring and calibration metrics
        temperature_model = ModelWithTemperature(model, device).to(device)
        val_metrics = temperature_model.calculate_metrics(devloader, train=True)
        for k, v in val_metrics.items():
            log_data[f'calibrate_{k}'] = v

        test_metrics = temperature_model.calculate_metrics(testloader)
        for k, v in test_metrics.items():
            clls[f'test_{k}'].append(v)
        temps.append(temperature_model.temperature.item())
    for metric in clls:
        log_data[metric] = np.mean(clls[metric])

    if log_metrics:
        logger.log(log_data)
        with open(f"{metrics_dir}/{filename}.json", 'wt') as f:
            f.write(json.dumps(log_data))

        # Write out test set logits and temperature
        np.save(f"{metrics_dir}/logits_{seed}.npy", np.array(eval_logits))
        #Save the temperature
        # with open(f"{metrics_dir}/temperature_{seed}.txt", 'wt') as f:
        #     for temp in temps:
        #         f.write(f"{temp}\n")


    if return_metrics:
        return log_data
