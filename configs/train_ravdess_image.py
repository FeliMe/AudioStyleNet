import os

from models import models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'train_path': HOME + '/Datasets/RAVDESS/Image/train',
    'val_path': HOME + '/Datasets/RAVDESS/Image/val',
    'data_format': 'image',

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Model parameters
    'model': models.PreTrainedResNet18(),

    # Logging
    'log_interval': 1000,
    'save_interval': 1,
    'save_path': 'saves/Classification_Landmarks'
})
