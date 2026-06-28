import torch
import pandas as pd
import gc
from scripts.single_training_run_cnnlstm import train_single_run
from src.config import DEFAULT_TRAINING_PARAMS_V1, DEFAULT_TRAINING_PARAMS_V2

def grid_search(search_space, device, experiment_root=None, model_version="1"):

    results = []

    if not experiment_root:
        raise ValueError("Please provide a valid path for the experiment_root parameter.")
    else:
        experiment_root.mkdir(parents=True, exist_ok=True)

    for run_id, params in enumerate(search_space, start=1):

        print(f"\nStarting run {run_id}/{len(search_space)}")
        print(params)

        run_name = (f"run_{run_id}")

        save_dir = experiment_root / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_version == "1":
            hyperparameters = DEFAULT_TRAINING_PARAMS_V1.copy()
        elif model_version == "2":
            hyperparameters = DEFAULT_TRAINING_PARAMS_V2.copy()
        else:
            raise(ValueError("incorrect model version passed"))
        
        hyperparameters.update(params)

        result = train_single_run(hyperparameters, device, save_dir, run_name, model_version=model_version)

        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_root / "grid_search_results.csv", index=False)
        
        gc.collect()

        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"CUDA cleanup warning: {e}")

    print("Grid search complete.")
