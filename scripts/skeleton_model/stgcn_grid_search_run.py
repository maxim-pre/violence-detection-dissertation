from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "test_batch_size"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "baseline_2"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "bs12", "batch_size": 12},
        {"run_name": "bs16", "batch_size": 16},
        {"run_name": "bs24", "batch_size": 24},
        {"run_name": "bs32", "batch_size": 32},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)