#%%

from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid, save_image

import dataloader
import models.discriminators as d
import models.generators as g
import models.model_utils as mutils
import utils


# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#%%

# Dataset path
HOME = os.path.expanduser('~')
dataroot = HOME + '/Datasets/celeba/'

# Hyperparameters
workers = 8
batch_size = 64
image_size = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5
nc = 3  # Number of channels
nz = 100  # Size of z latent vector
ngf = 64  # Size of feature maps in generator
ndf = 16  # # Size of feature maps in discriminator
sequence_length = 1
mean = [0.755, 0.673, 0.652]
std = [0.300, 0.348, 0.361]
normalize = False

#%%

ds = dataloader.RAVDESSDSPix2Pix(HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
                                 HOME + '/Datasets/RAVDESS/Image128',
                                 'image',
                                 use_same_sentence=True,
                                 normalize=normalize,
                                 mean=mean,
                                 std=std,
                                 max_samples=None,
                                 seed=manualSeed,
                                 sequence_length=sequence_length,
                                 step_size=1,
                                 image_size=64)

print("Found {} samples in total".format(len(ds)))

# Create the dataloader
data_loaders, dataset_sizes = dataloader.get_data_loaders(
    ds, 0.1, batch_size, True)

print("Using {} samples for training and {} for validation".format(
    dataset_sizes['train'], dataset_sizes['val']))


# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

print("Training on {}".format(device))

#%%

# Plot some training images
batch = next(iter(data_loaders['train']))
real_B = batch['B'][:, 0]
if normalize:
    transform = utils.denormalize(mean, std)
    real_B = torch.stack([transform(a) for a in real_B], 0).detach()
real_img = make_grid(real_B, padding=5, normalize=False)
os.makedirs('test_images', exist_ok=True)
save_image(real_img, 'test_images/Training_images.png')

#%%

# Init generator and discriminator
generator = g.SequenceGenerator(False, 0).to(device)
generator.apply(mutils.weights_init)

discriminator = d.SimpleDiscriminator(False).to(device)
discriminator.apply(mutils.weights_init)

#%%

# Initialize Loss function
criterionGAN = utils.GANLoss('vanilla', device)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, sequence_length, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

#%%

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for i_epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(data_loaders['train'], 0):
        real_A = data['A'].to(device)
        real_B = data['B'].to(device)

        # Forward
        fake_B = generator(real_A)

        ############################
        # (1) Train discriminator
        ###########################

        optimizerD.zero_grad()

        # Train with all-real batch
        pred_real = discriminator(real_B)
        loss_D_real = criterionGAN(pred_real, True, discriminator=True)

        # Train with all-fake batch
        pred_fake = discriminator(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, False, discriminator=True)

        loss_D_total = loss_D_real + loss_D_fake

        loss_D_real.backward()
        loss_D_fake.backward()

        # Update D
        optimizerD.step()

        ############################
        # (2) Train Generator
        ###########################
        optimizerG.zero_grad()
        pred_fake = discriminator(fake_B)
        loss_G = criterionGAN(pred_fake, True)
        loss_G.backward()

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (i_epoch + 1, num_epochs, i, len(data_loaders['train']),
                     loss_D_total.item(), loss_G.item()))

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D_total.item())

        iters += 1

    # Epoch finished
    if (i_epoch + 1) % 10 == 0:
        val_batch = next(iter(data_loaders['val']))
        with torch.no_grad():
            # Generate fake
            fake_B = generator(val_batch['A'].to(device))[:, 0].detach().cpu()

            # Denormalize
            if normalize:
                transform = utils.denormalize(mean, std)
                fake_B = torch.stack([transform(a) for a in fake_B], 0).detach()

            # Create grid image
            img = make_grid(fake_B, padding=5, normalize=False)

            # Save image
            save_image(img, 'test_images/sample_{}.png'.format(i_epoch + 1))

#%%

# Plot loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.show()
plt.savefig('test_images/losses.png')

#%%

# Real images vs fake images
batch = next(iter(data_loaders['val']))
real_B = batch['B'].to(device)
fake_B = generator(real_B)

real_B = real_B[:, 0]
fake_B = fake_B[:, 0]

# Denormalize
if normalize:
    transform = utils.denormalize(mean, std)
    real_B = torch.stack([transform(a) for a in real_B], 0).detach()
    fake_B = torch.stack([transform(a) for a in fake_B], 0).detach()

real_img = make_grid(real_B, padding=5, normalize=False)
fake_img = make_grid(fake_B, padding=5, normalize=False)

real_img = torch.nn.functional.pad(real_img, [0, 30, 0, 0], mode='constant')

# Cat real and fake together
imgs = torch.cat((real_img, fake_img), -1)
imgs = make_grid(imgs, nrow=1, normalize=False)

save_image(imgs, 'test_images/real_and_fake.png')
