import torch 
import pandas as pd
import gc
from matplotlib.pylab import rint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.common.train_one_epoch import train_one_epoch
from scripts.train_single_sepconvlstm import train_single_run
from scripts.common.evaluate import evaluate
from src.rwf2000 import RWF2000Dataset
from src.baseline_cnn_lstm import BaselineCNNLSTM
from src.baseline_cnn_lstm_2 import BaselineCNNLSTM2
from src.config import DATASET_ROOT
from src.config import CHECKPOINT_DIR
import json

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_name = "v1_truncated_cnn_64_frames"
    save_dir = CHECKPOINT_DIR / "baseline_cnn_SepConvLSTM" / run_name

    hyperparameters = {
        "num_frames": 32,
        "batch_size": 4,
        "epochs": 75,
        "augment": True,
        "freeze_cnn": True,
        "partial_freeze_cnn": False,
        "early_stopping_patience": 15,
        "scheduler_patience": 5,
        "hidden_channels": 128,
        "learning_rate": 1e-4,
        "dropout": 0.3,
        "reduced_channels": 64,
        "cnn_cutoff": 19,
        }

    train_single_run(hyperparameters, device, save_dir, run_name)

    