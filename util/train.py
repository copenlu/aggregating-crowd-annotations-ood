import torch
from tqdm import tqdm
from typing import AnyStr, List

from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from util.metrics import acc_f1
from util.model import AutoModelForMultiTaskSequenceClassification
from util.eval import evaluate

from util.loss import (CrossEntropyLoss,
                       KLLoss,
                       HardSampleLoss)


def train_cls(
        logger,
        model_name: AnyStr,
        trainloader,
        devloader,
        loss_class,
        multi_task: bool,
        device: torch.device,
        output_dir: AnyStr,
        weight_decay: float = 0.0,
        warmup_steps: int = 200,
        n_epochs: int = 3,
        train_majority: bool = False,
        class_weights: List = None,
        num_labels: int = 2,
        lr: float = 2e-5
):
    # Get the dataset
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if multi_task:
        model = AutoModelForMultiTaskSequenceClassification(model_name, config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    model = model.to(device)
    # Save an initial version of the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        n_epochs * len(trainloader)  # total
    )

    loss_fn = eval(loss_class)().to(device)
    best_F1 = -1.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, batch in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            batch = {k: v.to(device) for k, v in batch.items()}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            if multi_task:
                loss = loss_fn(
                    soft_labels=batch['bayes'],
                    labels=batch['majority_labels'] if train_majority else batch['labels'],
                    logits=outputs['logits'],
                    multi_task=True,
                    soft_logits=outputs['soft_logits'],
                    class_weights=class_weights
                )
            else:
                loss = loss_fn(
                    soft_labels=batch['bayes'],
                    labels=batch['majority_labels'] if train_majority else batch['labels'],
                    logits=outputs['logits'],
                    multi_task=False,
                    class_weights=class_weights
                )
            loss.backward()
            optimizer.step()
            scheduler.step()

            logger.log({
                'loss': loss.item()
            })
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        dev_labels, dev_logits, dev_preds = evaluate(devloader, model, device)
        acc, P, R, F1 = acc_f1(dev_logits, dev_labels, average='macro')
        if F1 > best_F1:
            print(f"Best validation F1: {F1}")
            best_F1 = F1
            torch.save(model.state_dict(), f'{output_dir}/model.pth')
            logger.log({
                'Val acc': acc,
                'Val P': P,
                'Val R': R,
                'Val F1': F1
            })

    model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    return model