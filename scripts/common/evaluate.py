import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * videos.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}"
            )
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc