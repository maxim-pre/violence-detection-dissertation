import torch 
from scripts.single_training_run_cnnlstm import train_single_run
from src.config import CHECKPOINT_DIR

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_version = "1"

    if model_version == '1':
        run_name = "test"
        save_dir = CHECKPOINT_DIR / "CNN_LSTM_V1" / run_name

        hyperparameters = {
            # optimizer hyperparameters
            "num_frames": 32,
            "augment": True,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "min_lr": 1e-5,
            "factor": 0.5,
            "scheduler_patience": 5,
            "epochs": 75,
            "early_stopping_patience": 15,

            # model hyperparameters
            "hidden_size": 64,
            "num_layers":1,
            "dropout": 0.35,
            "freeze_cnn": True,
            "cnn_cutoff": 16,
        }

        train_single_run(hyperparameters, device, save_dir, run_name, model_version=model_version)

    
    elif model_version == "2":

        run_name = "v1_updated_cropping_and_lr_schedule_64_channels"
        save_dir = CHECKPOINT_DIR / "CNN_LSTM_V2" / run_name

        hyperparameters = {
            # optimizer hyperparameters
            "num_frames": 32,
            "augment": True,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "min_lr": 1e-5,
            "factor": 0.5,
            "scheduler_patience": 5,
            "epochs": 75,
            "early_stopping_patience": 15,

            # model hyperparameters
            "hidden_channels": 64,
            "reduced_channels": 64,
            "dropout": 0.35,
            "freeze_cnn": True,
            "partial_freeze_cnn": False,
            "cnn_cutoff": 16,
        }

        train_single_run(hyperparameters, device, save_dir, run_name, model_version=model_version)

    