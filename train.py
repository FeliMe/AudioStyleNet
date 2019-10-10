import argparse
import importlib
import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

import dataloader

from solver import Solver

HOME = os.path.expanduser('~')
LOG_RUN = False


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

train_ds = dataloader.RAVDESSDataset(config.train_path,
                                     max_samples=None,
                                     format=config.data_format)
val_ds = dataloader.RAVDESSDataset(config.val_path,
                                   max_samples=None,
                                   format=config.data_format)

next(iter(train_ds))
1/0

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

data_loaders = {
    'train': train_loader,
    'val': val_loader
}

dataset_sizes = {
    'train': len(train_ds),
    'val': len(val_ds)
}

print("Found {} training and {} validation samples".format(
    len(train_ds), len(val_ds)))


""" Show data example """

x_sample, _ = next(iter(train_loader))
print('Input Shape: {}'.format(x_sample.shape))
# train_ds.show_sample()


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
