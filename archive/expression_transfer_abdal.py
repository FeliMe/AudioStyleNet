import argparse
import numpy as np
import os
import torch

from my_models.style_gan_2 import Generator
from torchvision.utils import save_image
from tqdm import tqdm


if __name__ == '__main__':
    """
    Expression transfer according to:
    https://www.researchgate.net/publication/332300501_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--source_neutral', type=str, required=True)
    parser.add_argument('-se', '--source_expr', type=str, required=True)
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('-d', '--save_dir', type=str,
                        default='saves/expression_transfer_abdal/')
    args = parser.parse_args()

    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load vectors
    source_neutral = torch.load(args.source_neutral).to(device)
    source_expression = torch.load(args.source_expr).to(device)
    target = torch.load(args.target).to(device)

    target_name = args.target.split('/')[-1].split('.')[0]
    expression_name = args.source_expr.split('/')[-1].split('.')[0]

    # Init generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Compute offset between expression and neutral pose
    offset = source_expression - source_neutral

    # Replace by 0 if L2-Norm is < 1.
    norm = torch.sqrt((offset ** 2).mean(dim=1)).unsqueeze(1)
    print(norm)
    offset = torch.where(norm > 1.1, offset, torch.zeros_like(offset))

    # Apply offset
    n_steps = 90

    min_offset = -0.2 * offset
    max_offset = 1.0 * offset

    os.makedirs(args.save_dir, exist_ok=True)
    for i in tqdm(range(n_steps)):
        w = min_offset + (i / n_steps) * (max_offset - min_offset)
        latent = target + w
        # latent = target + step * offset

        img_gen, _ = g([latent], input_is_latent=True, noise=g.noises)

        save_image(img_gen, args.save_dir + str(i + 1).zfill(3) +
                   '.png', normalize=True, range=(-1, 1))

    # Convert output frames to video
    original_dir = os.getcwd()
    os.chdir(args.save_dir)
    name = target_name + '_' + expression_name
    os.system(
        f'ffmpeg -framerate 30 -i %03d.png -c:v libx264 -r 30 -pix_fmt yuv420p {name}.mp4')

    # Remove generated frames and keep only video
    os.system('rm *.png')
    os.chdir(original_dir)

    # Generate source and target images
    # src_gen, _ = g([source_expression], input_is_latent=True, noise=g.noises)
    # save_image(src_gen, args.save_dir + f'{expression_name}.png', normalize=True, range=(-1, 1))
    # target_gen, _ = g([target], input_is_latent=True, noise=g.noises)
    # save_image(target_gen, args.save_dir + f'{target_name}.png', normalize=True, range=(-1, 1))
