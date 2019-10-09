import torch.nn as nn
import os

from models import models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'train_path': HOME + '/Datasets/RAVDESS/Landmarks/train',
    'val_path': HOME + '/Datasets/RAVDESS/Landmarks/val',
    'data_format': 'landmarks',

    # Training configs
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'log_interval': 1,
    'batch_size': 8,

    # Model configs
    'model': nn.Sequential(
        nn.Flatten(),
        nn.Linear(68 * 2, 8),
        nn.Softmax(dim=1)
    ),
})
