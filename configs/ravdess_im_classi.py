import os

from my_models import models
from utils.utils import Config

HOME = os.path.expanduser('~')

# ImageNet mean / std
IMG_NET_MEAN = [0.485, 0.456, 0.406]
IMG_NET_STD = [0.229, 0.224, 0.225]

# RAVDESS mean / std
RAVDESS_MEAN = [0.755, 0.673, 0.652]
RAVDESS_STD = [0.300, 0.348, 0.361]

# Universal mean / std
UNIVERSAL = [0.5, 0.5, 0.5]

# ImageNet mean / std (Grayscale)
RAVDESS_GRAY_MEAN = [0.694]
RAVDESS_GRAY_STD = [0.332]

config = Config({
    # General configs
    'use_cuda': True,
    'log_run': False,

    # Dataset configs
    # 'data_path': HOME + '/Datasets/RAVDESS/Aligned_256',
    'data_path': HOME + '/Datasets/RAVDESS/Image256',
    'data_format': 'image',
    'use_gray': False,
    'normalize': True,
    'validation_split': .2,
    'image_size': 256,
    'emotions': ['neutral', 'calm', 'happy', 'sad', 'angry',
                 'fearful', 'disgust', 'surprised'],
    'actors': [i + 1 for i in range(24)],

    # Hyper parameters
    'num_epochs': 5,
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

config.update({
    'mean': RAVDESS_GRAY_MEAN if config.use_gray else RAVDESS_MEAN,
    'std': RAVDESS_GRAY_STD if config.use_gray else RAVDESS_STD,
})
