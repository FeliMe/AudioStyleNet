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
    'random_seed': 123,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
    'target_data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'use_same_sentence': True,
    'validation_split': .1,
    'sequence_length': 5,
    'step_size': 1,
    'image_size': 64,
    'mean': [0.755, 0.673, 0.652],  # [0.694]
    'std': [0.300, 0.348, 0.361],  # [0.332]

    # Hyper parameters
    'num_epochs': 30,
    'lr_G': 0.0002,
    'lr_D': 0.0002,
    'batch_size': 32,
    'lambda_G_GAN': 1.,  # vanilla: noisy .5 not noisy
    'lambda_pixel': 10.,
    'lambda_emotion': .1,  # .1,

    # Loss functions
    'GAN_mode': 'vanilla',
    'criterion_pix': nn.L1Loss(),
    'criterion_emotion': nn.MSELoss(),

    # GAN hacks
    'noisy_labels': True,  # Use noisy labels for discriminator
    'label_range_real': (0.8, 1.1),
    'label_range_fake': (0.0, 0.2),
    'grad_clip_val': 0.0,  # Max gradient norm for discriminator, use 0 to disable grad clipping

    # Conditioning
    'num_conditioning_classes': 0,
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
    'optimizer_G': torch.optim.Adam(config.generator.parameters(),
                                    lr=config.lr_G, betas=(0.5, 0.999)),
    # 'optimizer_D': torch.optim.Adam(config.discriminator.parameters(),
    #                                 lr=config.lr_D,  betas=(0.5, 0.999)),
    'optimizer_D': torch.optim.SGD(config.discriminator.parameters(),
                                   lr=config.lr_D)
})
