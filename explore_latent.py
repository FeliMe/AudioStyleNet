import os
import torch

from projector import Projector
from my_models.generators import StyleGAN2Decoder
from PIL import Image
from torchvision import transforms

HOME = os.path.expanduser('~')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load target image
path = HOME + '/Datasets/RAVDESS/Image256/Actor_01/01-01-01-01-01-01-01/001.jpg'
target_image = Image.open(path)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])
target_image = transform(target_image)

# Load model
g = StyleGAN2Decoder().to(device).train()
for param in g.parameters():
    param.requires_grad = False

proj = Projector()

proj.set_network(g)
proj.run(target_image)
