import argparse
import importlib
import numpy as np
import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, RandomSampler
from torchsummaryX import summary

import dataloader

HOME = os.path.expanduser('~')
PLOT_GRADS = False
LOG_RUN = False

if PLOT_GRADS:
    print("WARNING: Plot gradients is on. This may cause slow training time!")


""" Load config """

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='ravdess_im_classi',
                    help='name of config')
args = parser.parse_args()

config = importlib.import_module('configs.' + args.config).config


""" Init wandb """

if LOG_RUN:
    wandb.init(project="emotion-pix2pix", config=config)


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)


""" Configure training with or without cuda """

if config.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    kwargs = {'pin_memory': True}
    print("GPU available. Training on {}.".format(device))
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}
    print("No GPU. Training on CPU.")


ds = dataloader.RAVDESSDSPix2Pix(config.data_path,
                                 data_format=config.data_format,
                                 use_gray=config.use_gray,
                                 max_samples=None,
                                 sequence_length=config.sequence_length,
                                 step_size=config.step_size)

sample = next(iter(ds))

print(sample['A'].shape, sample['B'].shape)
