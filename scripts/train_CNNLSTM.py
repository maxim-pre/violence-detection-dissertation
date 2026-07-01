import torch 
from scripts.single_training_run_cnnlstm import train_single_run
from src.config import CHECKPOINT_DIR, DEFAULT_TRAINING_PARAMS_V1, DEFAULT_TRAINING_PARAMS_V2

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_version = "2"


    if model_version == '1':
        run_name = "test"
        save_dir = CHECKPOINT_DIR / "CNN_LSTM_V1" / run_name
        params = {}
        hyperparameters = DEFAULT_TRAINING_PARAMS_V1.copy()
        hyperparameters.update(params)

        train_single_run(hyperparameters, device, save_dir, run_name, model_version=model_version)

    
    elif model_version == "2":

        run_name = "test_64_frames"
        save_dir = CHECKPOINT_DIR / "CNN_LSTM_V2" / run_name
        params = {"num_frames": 64}
        hyperparameters = DEFAULT_TRAINING_PARAMS_V2.copy()
        hyperparameters.update(params)

        train_single_run(hyperparameters, device, save_dir, run_name, model_version=model_version)

    