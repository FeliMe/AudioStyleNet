import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.utils import save_image, make_grid
from PIL import Image
from time import time

from my_models.generators import StyeGANDecoder
from utils import time_left
from perceptual_loss import PerceptualLoss


HOME = os.path.expanduser('~')
os.makedirs('saves/explore_latent', exist_ok=True)

iterations = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

target_image = Image.open(os.path.join(
    HOME, 'Datasets/RAVDESS/Image128/Actor_01/01-01-01-01-01-01-01/084.jpg'))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

target_image = transform(target_image).unsqueeze(0).to(device)

g = StyeGANDecoder(size=1024).to(device).train()

for param in g.parameters():
    param.requires_grad = False

# Use mean w over 10240 images as initial w
w = g.get_mean_w(device)

optimizer = torch.optim.Adam([w.requires_grad_()])
criterion = PerceptualLoss(net='vgg').to(device)

t_start = time()
for i in range(iterations):
    optimizer.zero_grad()

    y = g(w)
    y = F.interpolate(y, scale_factor=0.25)

    loss = criterion(y, target_image).view(-1)
    loss.backward()

    optimizer.step()

    if i % 100 == 0 and i != 0:
        t_left = time_left(t_start, iterations, i)
        print("[{}/{}], loss_l1: {:.4f}, time left: {}".format(
            i, iterations, loss.item(), t_left))
        sample = make_grid(torch.cat((y.detach().cpu(), target_image.cpu())))
        save_image(sample, "saves/explore_latent/sample_{}.jpg".format(i))
