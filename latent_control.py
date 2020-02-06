import argparse
import numpy as np
import os
import torch

from my_models.style_gan_2 import Generator
from torchvision.utils import save_image
from tqdm import tqdm


if __name__ == '__main__':

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_latent', type=str,
                        default='saves/projected_images/generated.pt')
    parser.add_argument('-v', '--vec', type=str,
                        default='saves/luxemburg_control_latent/smile.npy')
    parser.add_argument('-d', '--save_dir', type=str,
                        default='saves/luxemburg_control_latent/')
    args = parser.parse_args()

    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    latent_name = args.input_latent.split('/')[-1].split('.')[0]
    vec_type = args.vec.split('/')[-1].split('.')[0]

    # Init Generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Load vector
    vec = torch.tensor(np.load(args.vec),
                       dtype=torch.float32).unsqueeze(0).to(device)

    if args.input_latent == 'random':
        input_latent = g.get_latent(
            inp=[torch.randn(1, 512, device=device)],
            truncation=0.5,
            truncation_latent=g.latent_avg.to(device)
        )[0].unsqueeze(0)
        input_latent = input_latent.repeat(1, g.n_latent, 1)
    else:
        input_latent = torch.load(args.input_latent).unsqueeze(0).to(device)

    # Strengths of different vectors vary
    if vec_type == 'smile':
        min_mult = -4
        max_mult = 6
    elif vec_type == 'age':
        min_mult = -7
        max_mult = 11
    else:
        raise NotImplementedError

    min_vec = min_mult * vec
    max_vec = max_mult * vec

    n_steps = 120

    os.makedirs(args.save_dir, exist_ok=True)
    for i in tqdm(range(n_steps)):
        w = min_vec + (i / n_steps) * (max_vec - min_vec)
        with torch.no_grad():
            latent = input_latent + w
            img, _ = g([latent], input_is_latent=True, noise=g.noises)

            save_image(img, args.save_dir + str(i + 1).zfill(3) +
                       '.png', normalize=True, range=(-1, 1))

    # Convert output frames to video
    os.chdir(args.save_dir)
    name = latent_name + '_' + vec_type
    os.system(
        f'ffmpeg -framerate 30 -i %03d.png -c:v libx264 -r 30 -pix_fmt yuv420p {name}.mp4')

    # Remove generated frames and keep only video
    os.system('rm *.png')
