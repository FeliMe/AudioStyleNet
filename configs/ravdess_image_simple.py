import os
import torch
import torch.nn as nn

from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/Image',
    'data_format': 'image',
    'use_gray': True,
    'validation_split': .2,
    'sequence_length': 1,
    'window_size': 1,
    'step_size': 1,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Logging
    'log_interval': 1000,
    'save_interval': 1,
    'save_path': 'saves/Classification_Landmarks'
})

config.update({
    # Model parameters
    'model': nn.Sequential(
        nn.Conv2d(config.window_size, 16, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 112
        nn.Conv2d(16, 32, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 56
        nn.Conv2d(32, 64, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 28
        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 8),
    )
})

config.update({
    # Optimizer
    'optim': torch.optim.Adam(params=config.model.parameters(),
                              lr=config.learning_rate),
})
