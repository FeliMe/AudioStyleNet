import os
import torch

from models import models
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
    'sequence_length': 3,
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
    # 'model': models.ConvAndLSTM()
    # 'model': models.ConvAndCat(config.sequence_length)
    # 'model': models.ConvAndPool(config.sequence_length)
    'model': models.ConvAnd3D(config.sequence_length)
})

config.update({
    # Optimizer
    'optim': torch.optim.Adam(params=config.model.parameters(),
                              lr=config.learning_rate),
})
