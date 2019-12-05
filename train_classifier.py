import argparse
import importlib
import os
import torch
import wandb

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import dataloader

from solver_classifier import ClassificationSolver

HOME = os.path.expanduser('~')
PLOT_GRADS = False

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

if config.log_run:
    writer = SummaryWriter(
        'tensorboard_runs/img_classification/' + 
        datetime.now().strftime("%Y%m%d-%H%M%S"))
    wandb.init(project="emotion-classification", config=config)
else:
    writer = None


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)


""" Load dataset """

ds = dataloader.RAVDESSDataset(config.data_path,
                               data_format=config.data_format,
                               normalize=config.normalize,
                               mean=config.mean,
                               std=config.std,
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
print('Input Shape: {}'.format(sample['x'].shape))
# train_ds.plot_label_distribution()
# ds.show_sample()


""" Initialize model, solver, optimizer and criterion """

solver = ClassificationSolver(config)


""" Do training """

model = solver.train_model(data_loaders,
                           writer,
                           PLOT_GRADS)

solver.eval_model(data_loaders, writer)
