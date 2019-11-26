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
    'log_run': False,
    'random_seed': 999,
    'save_interval': 10,

    # Dataset configs
    'data_path': HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
    'target_data_path': HOME + '/Datasets/RAVDESS/Image128',
    'data_format': 'image',
    'use_gray': False,
    'normalize': False,
    'use_same_sentence': True,
    'validation_split': .1,
    'sequence_length': 1,
    'step_size': 1,
    'image_size': 64,
    'mean': [0.755, 0.673, 0.652],  # [0.694]
    'std': [0.300, 0.348, 0.361],  # [0.332]

    # Hyper parameters
    'num_epochs': 100,
    'lr_G': 0.0002,  # stable GAN: 0.0002
    'lr_D': 0.0002,  # stable GAN: 0.0002
    'batch_size': 64,
    'lambda_G_GAN': 1.,  # stable GAN: 1.
    'lambda_pixel': 0.,  # stable GAN: 10.
    'lambda_emotion': 0.,  # stable GAN: .1,

    # Loss functions
    'GAN_mode': 'vanilla',
    'criterion_pix': nn.L1Loss(),
    'criterion_emotion': nn.MSELoss(),

    # GAN hacks
    'noisy_labels': False,  # Use noisy labels for discriminator
    'label_range_real': (0.9, 1.0),  # stable GAN: (0.8, 1.1)
    'label_range_fake': (0.0, 0.2),  # stable GAN: (0.0, 0.2)
    'grad_clip_val': 0.0,  # Max gradient norm for discriminator, use 0 to disable grad clipping
    'flip_prob': 0.0,

    # Conditioning
    'n_classes_cond': 8,
})

config.update({
    # Models
    'generator': generators.SequenceGenerator(
        config.use_gray, config.n_classes_cond),
    # 'discriminator': discriminators.SequencePatchDiscriminator(
    #     config.use_gray, config.n_classes_cond),
    # 'discriminator': discriminators.SequenceDiscriminator(config.use_gray),
    'discriminator': discriminators.SequenceDiscriminator(config.use_gray,
                                                          config.n_classes_cond),

    # Classification model
    'classifier': models.ConvAndConvLSTM(config.use_gray),
    'classifier_path': 'saves/classifier_seq%d.pt' % int(config.sequence_length),
})

config.update({
    # Optimizers
    'optimizer_G': torch.optim.Adam(config.generator.parameters(),
                                    lr=config.lr_G, betas=(0.5, 0.999)),
    'optimizer_D': torch.optim.Adam(config.discriminator.parameters(),
                                    lr=config.lr_D,  betas=(0.5, 0.999)),
    # 'optimizer_D': torch.optim.SGD(config.discriminator.parameters(),
    #                                lr=config.lr_D)
})

config.update({
    'mean': RAVDESS_GRAY_MEAN if config.use_gray else RAVDESS_MEAN,
    'std': RAVDESS_GRAY_STD if config.use_gray else RAVDESS_STD,
})
