import torch
import pandas as pd
import gc
from src.Skeleton_model.stgcn import STGCN
from scripts.common.seed import set_seed
from src.config import DEFAULT_STGCN_PARAMS_V1, POSE_DATASET_ROOT
from scripts.common.get_device import get_available_device
from src.rwf2000 import RWF2000PoseDataset
from src.Skeleton_model.graph import SkeletonGraph, compute_joint_distance_to_center_of_gravity
from src.Skeleton_model.train_model import train_model



def grid_search(search_space, experiment_root=None):

    results = []

    if not experiment_root:
        raise ValueError("Please provide a valid path for the experiment_root parameter.")
    else:
        experiment_root.mkdir(parents=True, exist_ok=True)

    for run_id, params in enumerate(search_space, start=1):

        params = params.copy()
        run_name = params.pop("run_name", f"run_{run_id}")

        print(f"\nStarting run {run_id}/{len(search_space)}")
        print(params)

        save_dir = experiment_root / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        hyperparameters = DEFAULT_STGCN_PARAMS_V1.copy()
        hyperparameters.update(params)

        set_seed(hyperparameters["seed"])
        device = get_available_device()

        train_dataset = RWF2000PoseDataset(POSE_DATASET_ROOT, split="train", max_people=hyperparameters["max_people"])
        val_dataset = RWF2000PoseDataset(POSE_DATASET_ROOT, split="val", max_people=hyperparameters["max_people"])
        radii = compute_joint_distance_to_center_of_gravity(train_dataset)
        skeleton_graph = SkeletonGraph(radii, normalisation=hyperparameters["adjacency_normalisation_mode"])
        model = STGCN(skeleton_graph.A, temporal_kernel_size=hyperparameters["temporal_kernel_size"], dropout=hyperparameters["dropout"], edge_importance_weighting=hyperparameters["edge_importance_weighting"]).to(device)

        try:
            result = train_model(model, train_dataset, val_dataset, hyperparameters, device, save_dir, run_name)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM on {run_name}. Cleaning cache and retrying once...")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                device = get_available_device()
                model = STGCN(skeleton_graph.A, temporal_kernel_size=hyperparameters["temporal_kernel_size"], dropout=hyperparameters["dropout"], edge_importance_weighting=hyperparameters["edge_importance_weighting"]).to(device)
                result = train_model(model, train_dataset, val_dataset, hyperparameters, device, save_dir, run_name)
            except torch.cuda.OutOfMemoryError as e:
                print(f"out of memory again on {run_name}. Skipping run.")

                result = {
                    "run_name": run_name,
                    **hyperparameters,
                    "status": "OOM"
                }

        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_root / "grid_search_results.csv", index=False)

        del model
        del train_dataset
        del val_dataset
        del skeleton_graph
        gc.collect()

        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"CUDA cleanup warning: {e}")

    print("Grid search complete.")
