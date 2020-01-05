import os
import torch

from my_models import models
from utils.utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/Image',
    'data_format': 'image',
    'validation_split': .2,
    'sequence_length': 1,
    'step_size': 1,
    'image_size': 224,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Logging
    'log_interval': 1000,
    'save_interval': 1,
    'save_path': 'saves/Classification_Image'
})

config.update({
    # Model parameters
    'model': models.PreTrainedResNet18(config.window_size)
})

config.update({
    # Optimizer
    'optim': torch.optim.Adam(params=config.model.parameters(),
                              lr=config.learning_rate),
})
