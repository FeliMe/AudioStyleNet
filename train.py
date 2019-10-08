import os
import torch

from torch.utils.data import DataLoader, RandomSampler
from torchvision import models

import dataloader

HOME = os.path.expanduser('~')


""" Add a seed to have reproducible results """

seed = 123
torch.manual_seed(seed)

""" Configure training with or without cuda """

if torch.cuda.is_available():
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

train_ds = dataloader.RAVDESSDataset(HOME + '/Datasets/RAVDESS/Image/train')
val_ds = dataloader.RAVDESSDataset(HOME + '/Datasets/RAVDESS/Image/val')

train_sampler = RandomSampler(range(len(train_ds)))
val_sampler = RandomSampler(range(len(val_ds)))

train_loader = DataLoader(train_ds,
                          batch_size=8,
                          num_workers=8,
                          sampler=train_sampler,
                          drop_last=True)

val_loader = DataLoader(val_ds,
                        batch_size=8,
                        num_workers=8,
                        sampler=val_sampler,
                        drop_last=True)

""" Initialize model """

model = models.resnet18(pretrained=True)

x, y = next(iter(train_loader))

print(x.shape)
model(x)
