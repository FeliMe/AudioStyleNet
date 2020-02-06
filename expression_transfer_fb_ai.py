import argparse
import os
import torch

from my_models.style_gan_2 import Generator
from torchvision.utils import save_image


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('appearance', type=str)
    parser.add_argument('expression', type=str)
    parser.add_argument('--target_dir', type=str, default='saves/fb_expression_transfer/')
    args = parser.parse_args()

    appearance_name = args.appearance.split('/')[-1].split('.')[0]
    expression_name = args.expression.split('/')[-1].split('.')[0]

    # Specify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latents
    appearance = torch.load(args.appearance).to(device)
    expression = torch.load(args.expression).to(device)

    # Define mixing levels
    alpha = torch.ones((18, 1), device=device)
    alpha[3] = 0.
    alpha[4] = 0.

    beta = torch.ones_like(alpha) - alpha

    # Mix styles
    new = alpha * appearance + beta * expression

    # Init Generator
    g = Generator(1024, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Generated transferred image
    img_gen, _ = g([new], input_is_latent=True, noise=g.noises)
    appearance_gen, _ = g([appearance], input_is_latent=True, noise=g.noises)
    expression_gen, _ = g([expression], input_is_latent=True, noise=g.noises)

    # Save image
    os.makedirs(args.target_dir, exist_ok=True)
    save_appearance = os.path.join(
        args.target_dir, f"{appearance_name}-{expression_name}-appearance.png")
    save_expression = os.path.join(
        args.target_dir, f"{appearance_name}-{expression_name}-expression.png")
    save_gen = os.path.join(
        args.target_dir, f"{appearance_name}-{expression_name}-g.png")
    save_image(appearance_gen, save_appearance, normalize=True, range=(-1, 1))
    save_image(expression_gen, save_expression, normalize=True, range=(-1, 1))
    save_image(img_gen, save_gen, normalize=True, range=(-1, 1))
