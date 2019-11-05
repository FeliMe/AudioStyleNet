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
if 'use_gray' not in config.keys():
    config['use_gray'] = True


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
                               data_format=config.data_format,
                               use_gray=config.use_gray,
                               max_samples=None,
                               sequence_length=config.sequence_length,
                               step_size=config.step_size)

print("Found {} samples in total".format(len(ds)))

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

print("Using {} samples for training and {} for validation".format(
    len(train_indices), len(val_indices)))


""" Show data example """

x_sample, _ = next(iter(train_loader))
print('Input Shape: {}'.format(x_sample.shape))
# train_ds.plot_label_distribution()
# ds.show_sample()


""" Initialize model, solver, optimizer and criterion """

model = config.model
optimizer = config.optim
criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)

if LOG_RUN:
    wandb.watch(model)
model.train()
model.to(device)

solver = config.solver


print('Printing model summary...')
summary(model, torch.zeros((1, *x_sample.shape[1:])).to(device))


""" Do training """

model = solver.train_model(criterion,
                           optimizer,
                           device,
                           data_loaders,
                           dataset_sizes,
                           config,
                           exp_lr_scheduler,
                           PLOT_GRADS,
                           LOG_RUN)

if not solver.kill_now:
    solver.eval_model(device, data_loaders)
