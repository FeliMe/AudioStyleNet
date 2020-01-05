import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

from torchvision.utils import save_image, make_grid
from PIL import Image
from time import time

from my_models.generators import StyleGANDecoder
from utils.utils import time_left
from lpips import PerceptualLoss
from utils.perceptual_loss import VGG16Loss


def adjust_optim(optimizer, i_iter, initial_lr):
    # n_iter must start from 1
    if i_iter < 50:
        step = (0.1 - initial_lr) / 50
        new_lr = optimizer.param_groups[0]['lr'] + step
        optimizer.param_groups[0]['lr'] = new_lr
    elif i_iter == 50:
        optimizer.param_groups[0]['lr'] = 0.1
    elif i_iter > 750:
        step = (i_iter - 750) * ((math.pi / 2) / 250)
        new_lr = optimizer.param_groups[0]['lr'] * math.cos(step)


def add_noise_to_w(w, i_iter, std_w, device):
    if i_iter < 750:
        t = 1 - (i_iter / 750)
        noise = torch.randn(w.size()).to(device) * (0.005 * std_w * (t ** 2))
        w = w + noise
    return w


HOME = os.path.expanduser('~')
os.makedirs('saves/explore_latent', exist_ok=True)

iterations = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_image = Image.open(os.path.join(
    HOME, 'Datasets/RAVDESS/Image256/Actor_01/01-01-01-01-01-01-01/084.jpg'))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

target_image = transform(target_image).unsqueeze(0).to(device)

g = StyleGANDecoder(size=1024).to(device).train()

for param in g.parameters():
    param.requires_grad = False

# Use mean w over 10240 images as initial w
w, std_w = g.get_mean_std_w(device)

optimizer = torch.optim.Adam([w.requires_grad_()])
initial_lr = optimizer.param_groups[0]['lr']
criterion = PerceptualLoss(net='vgg').to(device)
# criterion = VGG16Loss(device)

t_start = time()
for i in range(iterations):
    optimizer.zero_grad()

    w = add_noise_to_w(w, i, std_w, device)

    y = g(w)
    y = F.interpolate(y, scale_factor=0.25)

    loss = criterion(y, target_image).view(-1)
    loss.backward()

    optimizer.step()

    adjust_optim(optimizer, i + 1, initial_lr)

    if (i % 100 == 0 and i != 0) or (i + 1 == iterations):
        t_left = time_left(t_start, iterations, i)
        print("[{}/{}], loss: {:.4f}, time left: {}".format(
            i, iterations, loss.item(), t_left))
        sample = make_grid(torch.cat((y.detach().cpu(), target_image.cpu())))
        save_image(sample, "saves/explore_latent/sample_{}.jpg".format(i))
