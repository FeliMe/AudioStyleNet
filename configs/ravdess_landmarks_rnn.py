import torch
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
    'sequence_length': 20,
    'window_size': 3,
    'step_size': 1,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.005,
    'batch_size': 32,

    # Logging
    'log_interval': 10000,
    'save_interval': 1,
    'save_path': 'saves/Classification_Landmarks'
})

config.update({
    # Model parameters
    'model': models.LandmarksLSTM(config.window_size, config.sequence_length)
})

config.update({
    # Optimizer
    'optim': torch.optim.SGD(config.model.parameters(), lr=config.learning_rate)
})
