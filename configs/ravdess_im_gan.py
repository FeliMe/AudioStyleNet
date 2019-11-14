import os
import torch.nn as nn

from models import gan_models, models
from utils import Config

HOME = os.path.expanduser('~')

config = Config({
    # General configs
    'use_cuda': True,
    'log_run': True,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
    'target_data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'validation_split': .1,
    'sequence_length': 9,
    'step_size': 1,
    'image_size': 64,
    'mean': [0.755, 0.673, 0.652],  # [0.694]
    'std': [0.300, 0.348, 0.361],  # [0.332]

    # Hyper parameters
    'num_epochs': 100,
    'lr_G': 0.0002,
    'lr_D': 0.0002,
    'batch_size': 32,
    'lambda_pixel': 10.,
    'lambda_emotion': .1,

    # Loss functions
    'GAN_mode': 'vanilla',
    'criterion_pix': nn.L1Loss(),
    'criterion_emotion': nn.MSELoss(),

    # Logging
    'save_interval': 1,
    'save_dir': 'saves/GAN',
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
