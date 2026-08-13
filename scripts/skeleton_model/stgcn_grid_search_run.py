from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "test_augmentation_2"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "initial_search"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "baseline"},

        {"run_name": "whole_occ_0.1", "augmentation_params": {"whole_occlusion_probability": 0.1}},
        {"run_name": "whole_occ_0.25", "augmentation_params": {"whole_occlusion_probability": 0.25}},

        {"run_name": "body_interp_0.1", "augmentation_params": {"body_part_interpolation_probability": 0.1}},
        {"run_name": "body_interp_0.25", "augmentation_params": {"body_part_interpolation_probability": 0.25}},
        {"run_name": "body_interp_0.5", "augmentation_params": {"body_part_interpolation_probability": 0.5}},

        {"run_name": "interp_0.1", "augmentation_params": {"interpolation_probability": 0.1}},
        {"run_name": "interp_0.25", "augmentation_params": {"interpolation_probability": 0.25}},
        {"run_name": "interp_0.5", "augmentation_params": {"interpolation_probability": 0.5}},

        {"run_name": "swap_0.1", "augmentation_params": {"swap_probability": 0.1}},
        {"run_name": "swap_0.25", "augmentation_params": {"swap_probability": 0.25}},
        {"run_name": "swap_0.5", "augmentation_params": {"swap_probability": 0.5}},

        {"run_name": "body_interp+swap", "augmentation_params": {"body_part_interpolation_probability": 0.25, "swap_probability": 0.25}},
        {"run_name": "interp+swap", "augmentation_params": {"interpolation_probability": 0.25, "swap_probability": 0.25}},
        {"run_name": "body_interp+interp", "augmentation_params": {"body_part_interpolation_probability": 0.25, "interpolation_probability": 0.25}},
        {"run_name": "whole_occ+body_interp", "augmentation_params": {"whole_occlusion_probability": 0.25, "body_part_interpolation_probability": 0.25}},
        {"run_name": "whole_occ+interp", "augmentation_params": {"whole_occlusion_probability": 0.25, "interpolation_probability": 0.25}},
        {"run_name": "whole_occ+swap", "augmentation_params": {"whole_occlusion_probability": 0.25, "swap_probability": 0.25}},

        {"run_name": "body_interp+interp+swap", "augmentation_params": {"body_part_interpolation_probability": 0.25, "interpolation_probability": 0.25, "swap_probability": 0.25}},
        {"run_name": "whole_occ+body_interp+swap", "augmentation_params": {"whole_occlusion_probability": 0.25, "body_part_interpolation_probability": 0.25, "swap_probability": 0.25}},
        {"run_name": "whole_occ+interp+swap", "augmentation_params": {"whole_occlusion_probability": 0.25, "interpolation_probability": 0.25, "swap_probability": 0.25}},
        {"run_name": "whole_occ+body_interp+interp", "augmentation_params": {"whole_occlusion_probability": 0.25, "body_part_interpolation_probability": 0.25, "interpolation_probability": 0.25}},

        {"run_name": "full_pipeline_0.1", "augmentation_params": {"whole_occlusion_probability": 0.1, "body_part_interpolation_probability": 0.1, "interpolation_probability": 0.1, "swap_probability": 0.1}},
        {"run_name": "full_pipeline_0.25", "augmentation_params": {"whole_occlusion_probability": 0.25, "body_part_interpolation_probability": 0.25, "interpolation_probability": 0.25, "swap_probability": 0.25}},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)