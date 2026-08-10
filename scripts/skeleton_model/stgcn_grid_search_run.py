from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_7_gamma_pose_no_augment"

    search_space = [
    {"run_name": "baseline_seed42", "seed": 42, "use_gamma_corrected_data": False, "augment": False},
    {"run_name": "baseline_seed43", "seed": 43, "use_gamma_corrected_data": False, "augment": False},
    {"run_name": "baseline_seed44", "seed": 44, "use_gamma_corrected_data": False, "augment": False},

    {"run_name": "gamma_seed42",    "seed": 42, "use_gamma_corrected_data": True, "augment": False},
    {"run_name": "gamma_seed43",    "seed": 43, "use_gamma_corrected_data": True, "augment": False},
    {"run_name": "gamma_seed44",    "seed": 44, "use_gamma_corrected_data": True, "augment": False},

    ]
    
    grid_search(search_space, experiment_root)