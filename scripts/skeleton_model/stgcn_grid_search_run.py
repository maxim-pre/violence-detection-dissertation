from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_5"

    search_space = [
        {"run_name": "no_aug",                  "augment": False, "augmentation_params": {}},
        {"run_name": "default_aug",             "augment": True,  "augmentation_params": {}},
        {"run_name": "short_occlusion",         "augment": True,  "augmentation_params": {"whole_occlusion_min_frames": 5,  "whole_occlusion_max_frames": 15}},
        {"run_name": "medium_occlusion",        "augment": True,  "augmentation_params": {"whole_occlusion_min_frames": 10, "whole_occlusion_max_frames": 25}},
        {"run_name": "long_occlusion",          "augment": True,  "augmentation_params": {"whole_occlusion_min_frames": 25, "whole_occlusion_max_frames": 50}},
        {"run_name": "occlusion_only",          "augment": True,  "augmentation_params": {"mirror_probability": 0.0}},
        {"run_name": "mirror_only",             "augment": True,  "augmentation_params": {"whole_occlusion_probability": 0.0}},
        {"run_name": "low_aug_probability",     "augment": True,  "augmentation_params": {"whole_occlusion_probability": 0.25, "mirror_probability": 0.25}},
        {"run_name": "high_aug_probability",    "augment": True,  "augmentation_params": {"whole_occlusion_probability": 0.75, "mirror_probability": 0.75}},
    ]
    
    grid_search(search_space, experiment_root)