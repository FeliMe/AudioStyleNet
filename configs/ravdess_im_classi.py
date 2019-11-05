import os
import torch

from models import models
from solver import ClassificationSolver
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
    'sequence_length': 9,
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
    # 'model': models.ConvAndCat(config.sequence_length)
    # 'model': models.ConvAndPool()
    # 'model': models.ConvAnd3D(config.sequence_length)
    # 'model': models.ConvAndRNN()
    'model': models.ConvAndConvLSTM()
    # 'model': models.SiameseConv3D()
    #
    # 'model': models.TestModel()
})

config.update({
    # Optimizer
    'solver': ClassificationSolver(config.model),
    'optim': torch.optim.Adam(params=config.model.parameters(),
                              lr=config.learning_rate),
})
