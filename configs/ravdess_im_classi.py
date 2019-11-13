import os

from models import models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,
    'log_run': False,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'validation_split': .2,
    'sequence_length': 5,
    'step_size': 1,
    'image_size': 64,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Logging
    'save_interval': 1,
    'save_path': 'saves/Classification_Image'
})

config.update({
    # Model parameters
    # 'model': models.ConvAndCat(config.sequence_length, config.use_gray)
    # 'model': models.ConvAndPool(config.use_gray)
    # 'model': models.ConvAnd3D(config.sequence_length, config.use_gray)
    # 'model': models.ConvAndRNN(config.use_gray)
    'model': models.ConvAndConvLSTM(config.use_gray)
    # 'model': models.SiameseConv3D(config.use_gray)
    #
    # 'model': models.TestModel()
})
