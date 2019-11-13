import os
import torch
import torch.nn as nn

from models import gan_models, models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,
    'log_run': False,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
    'target_data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'validation_split': .1,
    'sequence_length': 9,
    'step_size': 1,
    'image_size': 64,

    # Hyper parameters
    'num_epochs': 30,
    'lr_g': 0.0004,
    'lr_d': 0.0004,
    'batch_size': 32,
    'lambda_pixel': 50.,
    'lambda_emotion': 0.,

    # Logging
    'save_interval': 1,
    'save_path': 'saves/GAN',
})

config.update({
    # Models
    'generator': gan_models.SequenceGeneratorUNet(config.use_gray),
    'discriminator': gan_models.SequencePatchDiscriminator(config.use_gray),
    # 'discriminator': gan_models.SequenceDiscriminator(config.use_gray),

    # Classification model
    'classifier': models.ConvAndConvLSTM(config.use_gray),
    'classifier_path': 'saves/classifier_seq%d.pt' % int(config.sequence_length),
})

config.update({
    # Optimizers
    'optimizer_g': torch.optim.Adam(config.generator.parameters(),
                                    lr=config.lr_g, betas=(0.5, 0.999)),
    'optimizer_d': torch.optim.Adam(config.discriminator.parameters(),
                                    lr=config.lr_d, betas=(0.5, 0.999)),

    # Loss functions
    'criterion_gan': nn.BCEWithLogitsLoss(),
    'criterion_pix': nn.L1Loss(),
    'criterion_emotion': nn.MSELoss(),
})
