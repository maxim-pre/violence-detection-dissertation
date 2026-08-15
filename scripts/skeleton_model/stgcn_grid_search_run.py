from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "test_masked_batch_norm"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "baseline_2"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "test_masked_batch_norm"},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)