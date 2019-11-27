import os
import torch
import torch.nn as nn

from models import models, generators, discriminators
from utils import Config

HOME = os.path.expanduser('~')

# ImageNet mean / std
IMG_NET_MEAN = [0.485, 0.456, 0.406]
IMG_NET_STD = [0.229, 0.224, 0.225]

# RAVDESS mean / std
RAVDESS_MEAN = [0.755, 0.673, 0.652]
RAVDESS_STD = [0.300, 0.348, 0.361]

# ImageNet mean / std (Grayscale)
RAVDESS_GRAY_MEAN = [0.694]
RAVDESS_GRAY_STD = [0.332]

config = Config({
    # General configs
    'use_cuda': True,
    'log_run': True,
    'random_seed': 999,
    'save_interval': 10,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
    'target_data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'normalize': True,
    'use_same_sentence': True,
    'validation_split': .1,
    'sequence_length': 1,
    'step_size': 1,
    'image_size': 64,
    'mean': [0.755, 0.673, 0.652],  # [0.694]
    'std': [0.300, 0.348, 0.361],  # [0.332]

    # Model parameters
    'n_features_g': 64,
    'n_features_d': 64,

    # Hyper parameters
    'num_epochs': 30,
    'lr_G': 0.0002,  # stable GAN: 0.0002
    'lr_D': 0.0002,  # stable GAN: 0.0002
    'batch_size': 64,
    'lambda_G_GAN': 1.,
    'lambda_pixel': 100.,
    'lambda_emotion': 0.,

    # Loss functions
    'GAN_mode': 'vanilla',
    'criterion_pix': nn.L1Loss(),
    'criterion_emotion': nn.MSELoss(),

    # GAN hacks
    'noisy_labels': True,  # Use noisy labels for discriminator
    'label_range_real': (0.9, 1.0),
    'label_range_fake': (0.0, 0.2),
    'grad_clip_val': 0.0,  # Max gradient norm for discriminator, use 0 to disable grad clipping
    'flip_prob': 0.05,

    # Conditioning
    'n_classes_cond': 0,
})

config.update({
    # Generator
    'generator': generators.SequenceGenerator(
        config.use_gray,
        config.n_classes_cond,
        n_features=config.n_features_g
    ),

    # Discriminator
    'discriminator': discriminators.SequenceDiscriminator(
        config.use_gray,
        config.n_classes_cond,
        n_features=config.n_features_g
    ),
    # 'discriminator': discriminators.SequencePatchDiscriminator(
    #     config.use_gray,
    #     config.n_classes_cond
    # ),

    # Classification model
    'classifier': models.ConvAndConvLSTM(config.use_gray),
    'classifier_path': 'saves/classifier_seq%d.pt' % int(config.sequence_length),
})

config.update({
    # Optimizers
    'optimizer_G': torch.optim.Adam(
        config.generator.parameters(),
        lr=config.lr_G,
        betas=(0.5, 0.999)
    ),
    'optimizer_D': torch.optim.Adam(
        config.discriminator.parameters(),
        lr=config.lr_D,
        betas=(0.5, 0.999)
    ),
    # 'optimizer_D': torch.optim.SGD(
    #     config.discriminator.parameters(),
    #     lr=config.lr_D
    # )
})

config.update({
    # 'mean': RAVDESS_GRAY_MEAN if config.use_gray else RAVDESS_MEAN,
    # 'std': RAVDESS_GRAY_STD if config.use_gray else RAVDESS_STD,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
})
