from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS
from itertools import product


if __name__ == "__main__":
    model_version = "1"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "final_test_adjacency_norm_edge_importance"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "baseline_2"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "adj_column_edge_true",     "adjacency_normalisation_mode": "column",    "edge_importance_weighting": True},   
        {"run_name": "adj_column_edge_false",    "adjacency_normalisation_mode": "column",    "edge_importance_weighting": False},
        {"run_name": "adj_symmetric_edge_true",  "adjacency_normalisation_mode": "symmetric", "edge_importance_weighting": True},
        {"run_name": "adj_symmetric_edge_false", "adjacency_normalisation_mode": "symmetric", "edge_importance_weighting": False},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)