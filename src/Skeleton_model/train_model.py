import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import json
import pandas as pd
import gc
from scripts.common.seed import set_seed, seed_worker
from src.Skeleton_model.stgcn_train_one_epoch import train_one_epoch
from src.Skeleton_model.stgcn_eval_one_epoch import eval_one_epoch
from src.Skeleton_model.stgcn_build_optimizer import build_optimizer

def train_model(
        model, 
        train_dataset, 
        val_dataset,
        hyperparameters, 
        device, 
        save_dir,
        run_name="placeholder_name"
):
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(hyperparameters, f, indent=4)

    generator = torch.Generator()
    generator.manual_seed(hyperparameters["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True, 
        num_workers=4,
        pin_memory=True, 
        persistent_workers=True, 
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False, 
        num_workers=4,
        pin_memory=True, 
        persistent_workers=True, 
        worker_init_fn=seed_worker,
        generator=generator
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, hyperparameters)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=hyperparameters["factor"],
        patience=hyperparameters["scheduler_patience"], 
        min_lr=hyperparameters["min_lr"],
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(hyperparameters["epochs"]):
        print(f"\n{run_name} | Epoch {epoch+1}/{hyperparameters['epochs']}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)    
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

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
                "epoch": epoch+1, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history, 
                "config": hyperparameters, 
                "best_val_acc": best_val_acc,
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



            







