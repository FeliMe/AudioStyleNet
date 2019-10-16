import torch.nn as nn
import os

from models import models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/Landmarks',
    'data_format': 'landmarks',
    'validation_split': .2,
    'sequence_length': 3,
    'window_size': 3,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Logging
    'log_interval': 10000,
    'save_interval': 1,
    'save_path': 'saves/Classification_Landmarks'
})

config.update({
    # Model parameters
    # 'model': nn.Sequential(
    #     # nn.Conv1d(68 * 2, 68 * 2, config.sequence_length),
    #     # nn.ReLU(),
    #     nn.Flatten(),
    #     nn.Linear(68 * 2 * config.sequence_length, 8),
    #     # nn.ReLU(),
    #     # nn.Linear(512, 128),
    #     # nn.ReLU(),
    #     # nn.Linear(128, 8),
    # ),
    'model': models.LandmarksLSTM(config.window_size)
})
