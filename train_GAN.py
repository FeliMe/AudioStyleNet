import argparse
import importlib
import os
import torch
import wandb

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import dataloader

from solver_gan import GANSolver

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
torch.cuda.manual_seed(seed)


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

data_loaders, dataset_sizes = dataloader.get_data_loaders(
    ds, config.validation_split, config.batch_size, config.use_cuda)

print("Using {} samples for training and {} for validation".format(
    dataset_sizes['train'], dataset_sizes['val']))


""" Show data example """

sample = next(iter(data_loaders['train']))
print('Input Shape: {}'.format(sample['A'].shape))
# ds.show_sample()


""" Initialize solver """

solver = GANSolver(config)


""" Do training """

solver.train_model(data_loaders,
                   PLOT_GRADS,
                   writer)
