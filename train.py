import argparse
import importlib
import numpy as np
import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

import dataloader

from solver import Solver

HOME = os.path.expanduser('~')
LOG_RUN = True


""" Load config """

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='train_ravdess_landmarks',
                    help='name of config')
args = parser.parse_args()

config = importlib.import_module('configs.' + args.config).config


""" Init wandb """

if LOG_RUN:
    wandb.init(project="emotion-classification", config=config)


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

ds = dataloader.RAVDESSDataset(config.data_path,
                               max_samples=None,
                               sequence_length=config.sequence_length,
                               format=config.data_format)

# Split dataset
dataset_size = len(ds)
indices = list(range(dataset_size))
split = int(np.floor(config.validation_split * dataset_size))

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = RandomSampler(train_indices)
val_sampler = RandomSampler(val_indices)

train_loader = DataLoader(ds,
                          batch_size=config.batch_size,
                          num_workers=8,
                          sampler=train_sampler,
                          drop_last=True)

val_loader = DataLoader(ds,
                        batch_size=config.batch_size,
                        num_workers=8,
                        sampler=val_sampler,
                        drop_last=True)

data_loaders = {
    'train': train_loader,
    'val': val_loader
}

dataset_sizes = {
    'train': len(train_indices),
    'val': len(val_indices)
}

print("Found {} training and {} validation samples".format(
    len(train_indices), len(val_indices)))


""" Show data example """

x_sample, _ = next(iter(train_loader))
print('Input Shape: {}'.format(x_sample.shape))
# train_ds.plot_label_distribution()
# ds.show_sample()

""" Initialize model, solver, optimizer and criterion """

model = config.model
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)

if LOG_RUN:
    wandb.watch(model)
model.train()
model.to(device)

solver = Solver(model, LOG_RUN)

print('Printing model summary...')
print(summary(model, input_size=x_sample.shape[1:]))


""" Do training """

model = solver.train_model(criterion,
                           optimizer,
                           device,
                           data_loaders,
                           dataset_sizes,
                           config,
                           exp_lr_scheduler)

if not solver.kill_now:
    solver.eval_model(device, data_loaders)
