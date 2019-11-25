#%%

from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

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

#%%

# Create the dataset
# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))

ds = dataloader.RAVDESSDSPix2Pix(HOME + '/Datasets/RAVDESS/LandmarksLineImage128',
                                 HOME + '/Datasets/RAVDESS/Image128',
                                 'image',
                                 use_same_sentence=True,
                                 normalize=False,
                                 mean=[0.755, 0.673, 0.652],
                                 std=[0.300, 0.348, 0.361],
                                 max_samples=None,
                                 seed=123,
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
real_batch = next(iter(data_loaders['train']))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch['B'][:, 0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig("Training_images.png")


#%%

# Init generator and discriminator
generator = g.SequenceGenerator(False, 0).to(device)
generator.apply(mutils.weights_init)

discriminator = d.SimpleDiscriminator(False).to(device)
discriminator.apply(mutils.weights_init)

#%%

# Initialize BCELoss function
criterion = nn.BCELoss()

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
for epoch in range(num_epochs):
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
                  % (epoch, num_epochs, i, len(data_loaders['train']),
                     loss_D_total.item(), loss_G.item()))

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D_total.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(data_loaders['train'])-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            img_list.append(vutils.make_grid(fake[:, 0], padding=2, normalize=True))

        iters += 1

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
plt.savefig('losses.png')

#%%

# Show images
# fig = plt.figure(figsize=(8,v8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
# HTML(ani.to_jshtml())

#%%

# Real images vs fake images
real_batch = next(iter(data_loaders['val']))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch['B'][:, 0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
# plt.show()
plt.savefig('real_and_fake.png')
