import pandas as pd
import gc
from matplotlib.pylab import rint
import torch
import torch.nn as nn
from scripts.train_single_sepconvlstm import train_single_run
from src.config import CHECKPOINT_DIR

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

        result = train_single_run(hyperparameters, device, save_dir, run_name)

        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_root / "grid_search_results.csv", index=False)
        
        gc.collect()

        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"CUDA cleanup warning: {e}")

        print("Grid search complete.")


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_root = CHECKPOINT_DIR / "baseline_cnn_SepConvLSTM" / "grid_search_v2_2"

    search_space = [
    {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 64,  "dropout": 0.5, "learning_rate": 1e-5},
    {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 128, "reduced_channels": 64,  "dropout": 0.5, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 128, "dropout": 0.4, "learning_rate": 1e-5},
    {"hidden_channels": 64,  "reduced_channels": 128, "dropout": 0.5, "learning_rate": 1e-5},
    ]
    
    grid_search(search_space, device, experiment_root=experiment_root)