import pandas as pd
import gc
from matplotlib.pylab import rint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.common.train_one_epoch import train_one_epoch
from scripts.common.evaluate import evaluate
from src.rwf2000 import RWF2000Dataset
from src.baseline_cnn_lstm import BaselineCNNLSTM
from src.baseline_cnn_lstm_2 import BaselineCNNLSTM2
from src.config import DATASET_ROOT
from src.config import CHECKPOINT_DIR
import json

def grid_search(search_space, device, experiment_root=None):

    results = []

    if not experiment_root:
        raise ValueError("Please provide a valid path for the experiment_root parameter.")
    else:
        experiment_root.mkdir(parents=True, exist_ok=True)

    for run_id, params in enumerate(search_space, start=1):

        print(f"\nStarting run {run_id}/{len(search_space)}")
        print(params)

        run_name = (
            f"run_{run_id}_"
            f"hc{params['hidden_channels']}_"
            f"drop{params['dropout']}_"
            f"lr{params['learning_rate']}_"
        )

        save_dir = experiment_root / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        hyperparameters = {
            "num_frames": 32,
            "batch_size": 2,
            "epochs": 50,
            "augment": True,
            "partial_freeze_cnn": True,
            "early_stopping_patience": 10,
            "scheduler_patience": 3,
            "hidden_channels": params["hidden_channels"],
            "learning_rate": params["learning_rate"],
            "dropout": params["dropout"],
            "reduced_channels": params["reduced_channels"]
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(hyperparameters, f, indent=4)
        
        model = BaselineCNNLSTM2(
            hidden_channels=hyperparameters["hidden_channels"],
            dropout=hyperparameters["dropout"],
            freeze_cnn=not hyperparameters["partial_freeze_cnn"],
            partial_freeze_cnn=hyperparameters["partial_freeze_cnn"],
        ).to(device)

        train_dataset = RWF2000Dataset(DATASET_ROOT, split="train", num_frames=hyperparameters["num_frames"], augment=hyperparameters["augment"])
        val_dataset = RWF2000Dataset(DATASET_ROOT, split="val", num_frames=hyperparameters["num_frames"], augment=hyperparameters["augment"])

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
            lr=hyperparameters["learning_rate"]
        )


        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }


        best_val_acc = 0.0
        epochs_without_improvement = 0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

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
            "epochs_ran": len(history["train_acc"])
            }

        results.append(final_result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_root / "grid_search_results.csv", index=False)

        model = model.cpu()

        del model
        del optimizer
        del scheduler
        del criterion

        gc.collect()

        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"CUDA cleanup warning: {e}")

            print("Grid search complete.")


if __name__ == "__main__":

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "baseline_cnn_SepConvLSTM" / "grid_search_v2"

    search_space = [
    {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.5, "learning_rate": 1e-5},
    {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.5, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 128, "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 128, "dropout": 0.5, "learning_rate": 1e-5},
    ]
    
    grid_search(search_space, device, experiment_root=experiment_root)