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

    # Training configs
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'log_interval': 1,
    'batch_size': 8,

    # Model configs
    'model': models.landmark_model,
})
