from src.config import CHECKPOINT_DIR
from src.Skeleton_model.stgcn_grid_search import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "STGCN_V1" / "grid_search_3"

    search_space = [

    # ===== Best configuration (repeat with different seeds) =====
    {"run_name": "people3_seed42", "max_people": 3, "seed": 42},
    {"run_name": "people3_seed43", "max_people": 3, "seed": 43},
    {"run_name": "people3_seed44", "max_people": 3, "seed": 44},

    # ===== Test max_people = 4 =====
    {"run_name": "people4_seed42", "max_people": 4, "learning_rate": 2e-4, "dropout": 0.3},
    {"run_name": "people4_d04",    "max_people": 4, "learning_rate": 2e-4, "dropout": 0.4},
    {"run_name": "people4_lr15",   "max_people": 4, "learning_rate": 1.5e-4, "dropout": 0.3},
    {"run_name": "people4_lr25",   "max_people": 4, "learning_rate": 2.5e-4, "dropout": 0.3},

    # ===== Test max_people = 5 =====
    {"run_name": "people5_seed42", "max_people": 5, "learning_rate": 2e-4, "dropout": 0.3},
    {"run_name": "people5_d04",    "max_people": 5, "learning_rate": 2e-4, "dropout": 0.4},
    {"run_name": "people5_lr15",   "max_people": 5, "learning_rate": 1.5e-4, "dropout": 0.3},
    {"run_name": "people5_lr25",   "max_people": 5, "learning_rate": 2.5e-4, "dropout": 0.3},]
    
    grid_search(search_space, experiment_root)