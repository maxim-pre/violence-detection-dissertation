from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_1"

    search_space = [

        {"run_name": "baseline"},

        # Learning rate
        {"run_name": "lr_3e5", "learning_rate": 3e-5},
        {"run_name": "lr_5e5", "learning_rate": 5e-5},
        {"run_name": "lr_7e5", "learning_rate": 7e-5},
        {"run_name": "lr_2e4", "learning_rate": 2e-4},
        {"run_name": "lr_3e4", "learning_rate": 3e-4},

        # Weight decay
        {"run_name": "wd_1e5", "weight_decay": 1e-5},
        {"run_name": "wd_5e5", "weight_decay": 5e-5},
        {"run_name": "wd_1e4", "weight_decay": 1e-4},
        {"run_name": "wd_5e4", "weight_decay": 5e-4},
        {"run_name": "wd_1e3", "weight_decay": 1e-3},

        # Dropout
        {"run_name": "drop_02", "dropout": 0.2},
        {"run_name": "drop_03", "dropout": 0.3},
        {"run_name": "drop_04", "dropout": 0.4},
        {"run_name": "drop_06", "dropout": 0.6},
        {"run_name": "drop_07", "dropout": 0.7},

        # Adjacency normalisation
        {"run_name": "sym", "adjacency_normalisation_mode": "symmetric"},

        # Edge importance
        {"run_name": "no_edge", "edge_importance_weighting": False},

        # Combined regularisation
        {"run_name": "d03_wd1e4", "dropout": 0.3, "weight_decay": 1e-4},
        {"run_name": "d04_wd1e4", "dropout": 0.4, "weight_decay": 1e-4},
        {"run_name": "d03_wd5e4", "dropout": 0.3, "weight_decay": 5e-4},
        {"run_name": "d04_wd5e4", "dropout": 0.4, "weight_decay": 5e-4},

        # LR + Dropout
        {"run_name": "lr5e5_d03", "learning_rate": 5e-5, "dropout": 0.3},
        {"run_name": "lr5e5_d04", "learning_rate": 5e-5, "dropout": 0.4},
        {"run_name": "lr2e4_d03", "learning_rate": 2e-4, "dropout": 0.3},
        {"run_name": "lr2e4_d04", "learning_rate": 2e-4, "dropout": 0.4},

        # LR + Weight decay
        {"run_name": "lr5e5_wd1e4", "learning_rate": 5e-5, "weight_decay": 1e-4},
        {"run_name": "lr5e5_wd5e4", "learning_rate": 5e-5, "weight_decay": 5e-4},
        {"run_name": "lr2e4_wd1e4", "learning_rate": 2e-4, "weight_decay": 1e-4},
        {"run_name": "lr2e4_wd5e4", "learning_rate": 2e-4, "weight_decay": 5e-4},

        # Best guesses
        {"run_name": "best_1", "learning_rate": 5e-5, "dropout": 0.3, "weight_decay": 1e-4},
        {"run_name": "best_2", "learning_rate": 5e-5, "dropout": 0.4, "weight_decay": 1e-4},
        {"run_name": "best_3", "learning_rate": 1e-4, "dropout": 0.3, "weight_decay": 5e-4},
        {"run_name": "best_4", "learning_rate": 1e-4, "dropout": 0.4, "weight_decay": 5e-4},

        # Symmetric graph combinations
        {"run_name": "sym_d03", "adjacency_normalisation_mode": "symmetric", "dropout": 0.3},
        {"run_name": "sym_d04", "adjacency_normalisation_mode": "symmetric", "dropout": 0.4},
        {"run_name": "sym_best", "adjacency_normalisation_mode": "symmetric", "learning_rate": 5e-5, "dropout": 0.3, "weight_decay": 1e-4},

        # Max people
        {"run_name": "person1", "max_people": 1},
        {"run_name": "person1_best", "max_people": 1, "learning_rate": 5e-5, "dropout": 0.3, "weight_decay": 1e-4},

    ]
    
    grid_search(search_space, experiment_root)