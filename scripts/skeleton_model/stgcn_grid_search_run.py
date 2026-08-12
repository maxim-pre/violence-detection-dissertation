from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search
from src.config import DEFAULT_POSE_AUGMENTATION_PARAMS


if __name__ == "__main__":
    model_version = "2"  # Change to "2" for STGCNV2

    if model_version == "1":
        experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_7_combined_gamma_pose"
    elif model_version == "2":
        experiment_root = CHECKPOINT_DIR / "STGCN_V2" / "initial_search"
    else:
        raise ValueError("invalid model version")

    search_space = [
        {"run_name": "baseline"},

        # Learning rate
        {"run_name": "lr_5e-5", "learning_rate": 5e-5},
        {"run_name": "lr_1e-4", "learning_rate": 1e-4},
        {"run_name": "lr_3e-4", "learning_rate": 3e-4},
        {"run_name": "lr_5e-4", "learning_rate": 5e-4},

        # Dropout
        {"run_name": "dropout_0.2", "dropout": 0.2},
        {"run_name": "dropout_0.4", "dropout": 0.4},
        {"run_name": "dropout_0.5", "dropout": 0.5},
        {"run_name": "dropout_0.6", "dropout": 0.6},

        # Weight decay
        {"run_name": "wd_5e-5", "weight_decay": 5e-5},
        {"run_name": "wd_2e-4", "weight_decay": 2e-4},
        {"run_name": "wd_5e-4", "weight_decay": 5e-4},
        {"run_name": "wd_1e-3", "weight_decay": 1e-3},

        # Label smoothing
        {"run_name": "ls_0.05", "label_smoothing": 0.05},
        {"run_name": "ls_0.10", "label_smoothing": 0.10},
        {"run_name": "ls_0.15", "label_smoothing": 0.15},

        # Combined regularisation
        {"run_name": "reg_1", "dropout": 0.4, "weight_decay": 2e-4},
        {"run_name": "reg_2", "dropout": 0.4, "weight_decay": 5e-4},
        {"run_name": "reg_3", "dropout": 0.5, "weight_decay": 2e-4},
        {"run_name": "reg_4", "dropout": 0.5, "weight_decay": 5e-4},
        {"run_name": "reg_5", "learning_rate": 1e-4, "dropout": 0.4, "weight_decay": 2e-4},
        {"run_name": "reg_6", "learning_rate": 1e-4, "dropout": 0.5, "weight_decay": 5e-4},
        {"run_name": "reg_7", "learning_rate": 5e-5, "dropout": 0.5, "weight_decay": 1e-3},
        {"run_name": "reg_8", "learning_rate": 1e-4, "dropout": 0.4, "weight_decay": 5e-4, "label_smoothing": 0.05},
    ]
    
    grid_search(search_space, experiment_root, model_version=model_version)