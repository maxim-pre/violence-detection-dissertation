import pandas as pd
import gc
from matplotlib.pylab import rint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.common.train_one_epoch import train_one_epoch
from scripts.common.evaluate import evaluate
from src.rwf2000 import RWF2000Dataset
from cnn_lstm_v1 import CNNLSTMV1
from cnn_lstm_v2 import CNNLSTMV2
from src.config import DATASET_ROOT
import json

def train_single_run(hyperparameters, device, save_dir, run_name, model_version="v1"):
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    if model_version == 1:
        model = CNNLSTMV1(
            hidden_size=hyperparameters["hidden_size"],
            num_layers=hyperparameters["num_layers"], 
            dropout=hyperparameters["dropout"],
            freeze_cnn=hyperparameters["freeze_cnn"]
        ).to(device)
    
    elif model_version == 2:
        model = CNNLSTMV2(
            hidden_channels=hyperparameters["hidden_channels"],
            reduced_channels=hyperparameters["reduced_channels"],
            dropout=hyperparameters["dropout"],
            freeze_cnn=not hyperparameters["partial_freeze_cnn"],
            partial_freeze_cnn=hyperparameters["partial_freeze_cnn"],
            cnn_cutoff=hyperparameters["cnn_cutoff"]
        ).to(device)
        
    else:
        raise(ValueError("model version doesn't exist"))
    

    train_dataset = RWF2000Dataset(DATASET_ROOT, split="train", num_frames=hyperparameters["num_frames"], augment=hyperparameters["augment"])
    val_dataset = RWF2000Dataset(DATASET_ROOT, split="val", num_frames=hyperparameters["num_frames"], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        num_workers=4
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparameters["learning_rate"],
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=hyperparameters["scheduler_patience"],
        min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(hyperparameters["epochs"]):

        print(f"\n{run_name} | Epoch {epoch+1}/{hyperparameters['epochs']}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)


        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)    
        history["val_acc"].append(val_acc)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            checkpoint = {
                "run_name": run_name,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": hyperparameters,
                "best_val_acc": best_val_acc
            }

            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"new best model saved. Val Acc: {best_val_acc:.4f}")

        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= hyperparameters["early_stopping_patience"]:
            print("Early stopping triggered.")
            break
    
    final_result = {
        "run_name": run_name,
        **hyperparameters,
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_ran": len(history["train_acc"]), 
        }

    # Save full training history
    history_df = pd.DataFrame(history)
    history_df.index.name = "epoch"
    history_df.to_csv(save_dir / "history.csv")
    
    pd.DataFrame([final_result]).to_csv(save_dir / "summary.csv", index=False)

    model = model.cpu()
    del model, optimizer, scheduler, criterion
    gc.collect()

    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"CUDA cleanup warning: {e}")
        
    return final_result