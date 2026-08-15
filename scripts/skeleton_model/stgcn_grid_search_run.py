from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "people_aggregation_new_DS_2_aug"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "baseline_2"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "masked_mean_42", "people_aggregation": "masked_mean"},
        {"run_name": "masked_mean_43", "people_aggregation": "masked_mean", "seed":43},
        {"run_name": "masked_mean_44", "people_aggregation": "masked_mean", "seed":44},
        {"run_name": "masked_mean_45", "people_aggregation": "masked_mean", "seed":45},

        {"run_name": "max_42", "people_aggregation": "max"},
        {"run_name": "max_43", "people_aggregation": "max", "seed":43},
        {"run_name": "max_44", "people_aggregation": "max", "seed":44},
        {"run_name": "max_45", "people_aggregation": "max", "seed":45},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)