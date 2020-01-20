import argparse
import glob
import os
import torch

from my_models.style_gan_2 import Generator
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
    latent_path = args.latent
    if os.path.isdir(latent_path):
        if latent_path[-1] != '/':
            latent_path += '/'
        latents = sorted(glob.glob(latent_path + '*.pt'))
        sentence = latent_path.split('/')[-2]
    else:
        latents = [latent_path]
        sentence = 'single_frames'

    offset = torch.load(args.offset).to(device)

    # Load generator
    g = Generator(1024, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Get necessary paths
    base_dir_original = 'saves/offsets/{}_original/'.format(sentence)
    base_dir_applied = 'saves/offsets/{}_applied/'.format(sentence)
    os.makedirs(base_dir_original, exist_ok=True)
    os.makedirs(base_dir_applied, exist_ok=True)

    for i, latent_str in enumerate(latents):
        # Generate original and manipulated image
        latent = torch.load(latent_str).to(device)
        img_orig, _ = g([latent], input_is_latent=True, noise=g.noises)
        img_new, _ = g([latent + offset[min(len(latents), i)]], input_is_latent=True, noise=g.noises)

        # Save
        img_name = latent_str.split('/')[-1].split('.')[0]
        save_image(img_orig, base_dir_original + img_name + '.png', normalize=True)
        save_image(img_new, base_dir_applied + img_name + '.png', normalize=True)
