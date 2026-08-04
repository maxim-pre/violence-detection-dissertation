
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def freeze_cnn_batchnorm(model):
    for module in model.cnn.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

def train_one_epoch(model, dataloader, criterion, optimizer, device, hyperparameters):
    model.train()
    
    if hyperparameters["freeze_cnn_batchnorm"]:
        freeze_cnn_batchnorm(model)

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(videos)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

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