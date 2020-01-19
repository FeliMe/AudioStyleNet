import argparse
import os
import torch

from ..my_models.style_gan_2 import Generator
from torchvision.utils import save_image


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent', type=str, required=True)
    parser.add_argument('-o', '--offset', type=str, required=True)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latent and offset
    latent = torch.load(args.latent).to(device)
    offset = torch.load(args.offset).to(device)

    # Load generator
    g = Generator(1024, 512, 8, pretrained=True)
    g.noises = [n.to(device) for n in g.noises]

    # Generate original and manipulated image
    img_orig, _ = g([latent], input_is_latent=True, noise=g.noises)
    img_new, _ = g([latent + offset], input_is_latent=True, noise=g.noises)

    # Save
    img_name = args.latent.split('/')[-1].split('.')[0]
    base_dir = '../saves/offsets/applied/'
    os.makedirs(base_dir, exist_ok=True)
    save_image(img_orig, '{}original_{}.png'.format(base_dir, img_name))
    save_image(img_new, '{}applied_{}.png'.format(base_dir, img_name))
