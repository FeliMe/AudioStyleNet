import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

import configs
import dataloader
import utils

HOME = os.path.expanduser('~')


""" Load config """

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='train_ravdess_landmarks',
                    help='name of config')
args = parser.parse_args()

config = importlib.import_module('configs.' + args.config).config


""" Init wandb """

# wandb.init(project="emotion-aware-facial-animation")


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

""" Load dataset """

train_ds = dataloader.RAVDESSDataset(config.train_path,
                                     format=config.data_format)
val_ds = dataloader.RAVDESSDataset(config.val_path,
                                   format=config.data_format)

train_sampler = RandomSampler(range(len(train_ds)))
val_sampler = RandomSampler(range(len(val_ds)))

train_loader = DataLoader(train_ds,
                          batch_size=config.batch_size,
                          num_workers=8,
                          sampler=train_sampler,
                          drop_last=True)

val_loader = DataLoader(val_ds,
                        batch_size=config.batch_size,
                        num_workers=8,
                        sampler=val_sampler,
                        drop_last=True)


""" Show data example """

x_sample, _ = next(iter(train_loader))
print(x_sample.shape)
# train_ds.show_sample()

""" Initialize model """

model = config.model

print('Printing model summary...')
print(summary(model, input_size=x_sample.shape[1:]))


""" Training """

y_ = model(x_sample)
print(y_.shape)
