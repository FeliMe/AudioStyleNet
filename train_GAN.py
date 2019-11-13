import argparse
import importlib
import numpy as np
import os
import torch
import wandb

from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import dataloader
import utils

from models.gan_models import weights_init
from solver import GANSolver

HOME = os.path.expanduser('~')
PLOT_GRADS = False

if PLOT_GRADS:
    print("WARNING: Plot gradients is on. This may cause slow training time!")


""" Load config """

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='ravdess_im_gan',
                    help='name of config')
args = parser.parse_args()

config = importlib.import_module('configs.' + args.config).config


""" Init wandb """

if config.log_run:
    writer = SummaryWriter(
        'tensorboard_runs/pix2pix/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    wandb.init(project="emotion-pix2pix", config=config, sync_tensorboard=True)
else:
    writer = None


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

ds = dataloader.RAVDESSDSPix2Pix(config.data_path,
                                 config.target_data_path,
                                 data_format=config.data_format,
                                 use_gray=config.use_gray,
                                 max_samples=None,
                                 sequence_length=config.sequence_length,
                                 step_size=config.step_size,
                                 image_size=config.image_size)

print("Found {} samples in total".format(len(ds)))


""" Split dataset"""

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


""" Show data example """

sample = next(iter(train_loader))
print('Input Shape: {}'.format(sample['A'].shape))
# ds.show_sample()


""" Initialize models, solver, optimizers and loss functions """

# Initialize models
generator = config.generator
generator.train()
generator.to(device)
generator.apply(weights_init)

discriminator = config.discriminator
discriminator.train()
discriminator.to(device)
discriminator.apply(weights_init)

classifier = config.classifier
classifier.load_state_dict(torch.load(config.classifier_path, map_location=device))
classifier.train()
classifier.to(device)
for param in classifier.parameters():
    param.requires_grad = False

if LOG_RUN:
    wandb.watch(generator)
    wandb.watch(discriminator)

# Initialize Loss functions
criterion_gan = config.criterion_gan
criterion_pix = config.criterion_pix
criterion_emotion = config.criterion_emotion

# Initialize optimizers
optimizer_g = config.optimizer_g
optimizer_d = config.optimizer_d

# Initialize solver
solver = GANSolver(generator, discriminator, classifier, ds.mean, ds.std)

print("Generator: # params {} (trainable {})".format(
    utils.count_params(generator),
    utils.count_trainable_params(generator)
))
print("Discriminator: # params {} (trainable {})".format(
    utils.count_params(discriminator),
    utils.count_trainable_params(discriminator)
))


""" Do training """

solver.train_model(optimizer_g,
                   optimizer_d,
                   criterion_gan,
                   criterion_pix,
                   criterion_emotion,
                   device,
                   train_loader,
                   val_loader,
                   config,
                   PLOT_GRADS,
                   LOG_RUN,
                   writer)
