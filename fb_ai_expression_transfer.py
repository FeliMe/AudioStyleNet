import argparse
import torch

from my_models.style_gan_2 import Generator
from torchvision.utils import save_image


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('target', type=str)
    args = parser.parse_args()

    # Specify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latents
    src = torch.load(args.src).to(device)
    target = torch.load(args.target).to(device)

    # Define mixin levels
    alpha = torch.zeros((18, 1), device=device)
    alpha[3] = 1.
    alpha[4] = 1.

    beta = torch.ones_like(alpha) - alpha

    # Mix styles
    new = alpha * src + beta * target

    # Init Generator
    g = Generator(1025, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Generated transferred image
    img_gen, _ = g([new], input_is_latent=True, noise=g.noises)

    # Save image
    save_image(img_gen, 'saves/tttt.png', normalize=True)
