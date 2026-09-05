from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "final_seed_search_no_aug_2"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "baseline_2"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "final_seed42", "seed": 42, "augment":False},
        {"run_name": "final_seed43", "seed": 43, "augment":False},
        {"run_name": "final_seed44", "seed": 44, "augment":False},
        {"run_name": "final_seed45", "seed": 45, "augment":False},
        {"run_name": "final_seed46", "seed": 46, "augment":False},
        {"run_name": "final_seed47", "seed": 47, "augment":False},
        {"run_name": "final_seed48", "seed": 48, "augment":False},
        {"run_name": "final_seed49", "seed": 49, "augment":False},
        {"run_name": "final_seed50", "seed": 50, "augment":False},
        {"run_name": "final_seed51", "seed": 51, "augment":False},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)