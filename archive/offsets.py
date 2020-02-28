import argparse
import os
import pathlib
import torch

from glob import glob
from my_models.style_gan_2 import Generator
from torchvision.utils import save_image


def compute_offset(args):
    assert len(args.emotions) == 2
    emotion1 = args.emotions[0]
    emotion2 = args.emotions[1]
    mapping = {
        'neutral': '01',
        'calm': '02',
        'happy': '03',
        'sad': '04',
        'angry': '05',
        'fearful': '06',
        'disgust': '07',
        'surprised': '08'
    }

    actor = args.actor.split('/')[-1]
    print("Computing offset between {} and {} of {}".format(
        emotion1, emotion2, actor))

    # Find sentences
    sentences = [str(f) for f in list(pathlib.Path(args.actor).glob('*'))]
    sentences = sorted(sentences)

    sentences1 = list(filter(lambda s: s.split(
        '/')[-1].split('-')[2] == mapping[emotion1], sentences))
    sentences2 = list(filter(lambda s: s.split(
        '/')[-1].split('-')[2] == mapping[emotion2], sentences))

    print("Sentences 1:")
    for s in sentences1:
        print(s)

    print("Sentences 2:")
    for s in sentences2:
        print(s)

    # Get frames
    print("\n\n\n")
    print(sentences1[-1])
    frames1 = [str(f) for f in list(pathlib.Path(sentences1[-1]).glob('*'))]

    frames2 = []
    for sentence in sentences2:
        for f in list(pathlib.Path(sentence).glob('*')):
            frames2.append(str(f))

    frames1 = sorted(frames1)
    frames2 = sorted(frames2)

    # Load latent vectors
    vectors1 = torch.stack([torch.load(frame) for frame in frames1])
    vectors2 = torch.stack([torch.load(frame) for frame in frames2])

    # Compute mean offset
    offset_means = []
    for vector in vectors1:
        offset_means.append((vectors2 - vector).mean(dim=0))
    offset_means = torch.stack(offset_means)
    print(offset_means.shape)
    offset = offset_means

    # Save
    os.makedirs(args.target_dir, exist_ok=True)
    torch.save(offset, '{}{}_sentence-1_offset_{}-{}.pt'.format(
        args.target_dir, actor, emotion1, emotion2
    ))


def apply_offset(args):
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latent and offset
    latent_path = args.latent
    if os.path.isdir(latent_path):
        if latent_path[-1] != '/':
            latent_path += '/'
        latents = sorted(glob(latent_path + '*.pt'))
        sentence = latent_path.split('/')[-2]
    else:
        latents = [latent_path]
        sentence = 'single_frames'

    offset = torch.load(args.offset).to(device)

    # Load generator
    g = Generator(1024, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    # Get necessary paths
    base_dir_original = '{}{}_original/'.format(args.target_dir, sentence)
    base_dir_applied = '{}{}_applied/'.format(args.target_dir, sentence)
    os.makedirs(base_dir_original, exist_ok=True)
    os.makedirs(base_dir_applied, exist_ok=True)

    for i, latent_str in enumerate(latents):
        # Generate original and manipulated image
        latent = torch.load(latent_str).to(device)
        img_orig, _ = g([latent], input_is_latent=True, noise=g.noises)
        img_new, _ = g([latent + offset[min(len(latents), i)]],
                       input_is_latent=True, noise=g.noises)

        # Save
        img_name = latent_str.split('/')[-1].split('.')[0]
        save_image(img_orig, base_dir_original +
                   img_name + '.png', normalize=True)
        save_image(img_new, base_dir_applied +
                   img_name + '.png', normalize=True)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_dir', type=str, default='saves/offsets/')

    # Args for compute offset
    parser.add_argument('-a', '--actor', type=str)
    parser.add_argument('-e', '--emotions', nargs='+',
                        help='two emotions to calculate the offset vector between')

    # Args for apply offset
    parser.add_argument('-l', '--latent', type=str)
    parser.add_argument('-o', '--offset', type=str)
    args = parser.parse_args()

    # Correct paths
    if args.actor[-1] == '/':
        args.actor = args.actor[:-1]
    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    
