import torch
from src.config import CHECKPOINT_DIR
from scripts.gridsearch_CNNLSTM import grid_search


if __name__ == "__main__":

    experiment_root = CHECKPOINT_DIR / "CNN_LSTM_V3" / "grid_search_3_seedtesting"

    search_space = [
        {"run_name": "seed_42", "seed": 42},
        {"run_name": "seed_43", "seed": 43},
        {"run_name": "seed_44", "seed": 44},
        {"run_name": "seed_42_classifier64", "classifier_hidden_size": 64, "seed": 42},
        {"run_name": "seed_43_classifier64", "classifier_hidden_size": 64, "seed": 43},
        {"run_name": "seed_44_classifier64", "classifier_hidden_size": 64, "seed": 44},
        {"run_name": "seed_42_lr75e5", "learning_rate": 7.5e-5, "seed": 42},
        {"run_name": "seed_43_lr75e5", "learning_rate": 7.5e-5, "seed": 43},
        {"run_name": "seed_44_lr75e5", "learning_rate": 7.5e-5, "seed": 44},
    ]
    
    grid_search(search_space, experiment_root=experiment_root, model_version="3")