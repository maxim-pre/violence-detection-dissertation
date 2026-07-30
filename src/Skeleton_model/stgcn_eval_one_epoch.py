
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0 
    correct = 0 
    total = 0 

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():

        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total 
    return epoch_loss, epoch_acc