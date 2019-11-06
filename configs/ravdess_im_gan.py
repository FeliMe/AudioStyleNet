import os
import torch

from models import gan_models
from solver import ClassificationSolver
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'validation_split': .2,
    'sequence_length': 9,
    'step_size': 1,
    'image_size': 64,

    # Hyper parameters
    'num_epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 32,

    # Logging
    'log_interval': 1000,
    'save_interval': 1,
    'save_path': 'saves/GAN'
})

config.update({
    # Models
    'generator': gan_models.SequenceGeneratorUNet(config.use_gray),
    'discriminator': gan_models.SequenceDiscriminator(config.sequence_length,
                                                      config.use_gray),
})

config.update({
    # Optimizers
    'optimizer_G': torch.optim.Adam(params=config.generator.parameters(),
                                    lr=config.learning_rate),
})
