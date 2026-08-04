from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS, DEFAULT_STRONG_POSE_AUGMENTATION_PARAMS


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_4"

    search_space = [
            {"run_name": "aug_current", "label_smoothing": 0.0, "augmentation_params": DEFAULT_POSE_AUGMENTATION_PARAMS, "augment": True},
            {"run_name": "aug_stronger", "label_smoothing": 0.0, "augmentation_params": DEFAULT_STRONG_POSE_AUGMENTATION_PARAMS, "augment": True},
            {"run_name": "aug_current_ls01", "label_smoothing": 0.1, "augmentation_params": DEFAULT_POSE_AUGMENTATION_PARAMS, "augment": True},
            {"run_name": "aug_stronger_ls01", "label_smoothing": 0.1, "augmentation_params": DEFAULT_STRONG_POSE_AUGMENTATION_PARAMS, "augment": True},
            {"run_name": "no_aug", "augment": False},
            {"run_name": "no_aug_ls01", "label_smoothing": 0.1, "augment": False},
    ]
    
    grid_search(search_space, experiment_root)