"""
File to specify dataloaders for different datasets
"""

import numpy as np
import random
import torch

from glob import glob
from my_models.style_gan_2 import Generator
from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from utils.utils import downsample_256


class Downsample(object):
    """ Custom transform: Downsamples image in StyleGAN2 manner """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        c, h, w = sample.shape
        if h > self.size:
            factor = h // self.size
            sample = sample.reshape(
                c, h // factor, factor, w // factor, factor)
            sample = sample.mean([2, 4])
        return sample


class ImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256):
        super().__init__()
        self.normalize = normalize
        self.mean = mean
        self.std = std

        print(f"Searching data in {root_path}")
        self.paths = sorted(glob(root_path + '*/*.png'))
        assert len(self.paths) > 0, "ImageDataset is empty"

        random.shuffle(self.paths)

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.t(Image.open(self.paths[index]))
        return {'img': img}


class AudioVisualDataset(Dataset):
    def __init__(self,
                 paths,
                 audio_type='deepspeech',
                 load_img=True,
                 load_latent=False,
                 random_inp_latent=False,
                 T=8,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 len_dataset=None):
        super().__init__()
        self.audio_type = audio_type
        self.load_img = load_img
        self.load_latent = load_latent
        self.random_inp_latent = random_inp_latent
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.T = T
        self.len_dataset = len_dataset

        self.paths = [item for sublist in paths for item in sublist]

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return self.len_dataset if self.len_dataset else len(self.paths)

    def __getitem__(self, indices):
        paths = [self.paths[i] for i in indices]
        audio_paths = paths[:-1]
        input_path = paths[-1]
        target_path = paths[self.T // 2]

        video = '/'.join(input_path.split('/')[:-1]) + '/'

        # Load audio
        audio = []
        for p in audio_paths:
            audio.append(torch.tensor(
                np.load(f"{p}.{self.audio_type}.npy"), dtype=torch.float32)[:, :32])
        audio = torch.stack(audio, dim=0)

        # Load images
        if self.load_img:
            target_img = self.t(Image.open(target_path + '.png'))
        else:
            target_img = torch.tensor(0.)

        # Load latents
        if self.load_latent:
            if self.random_inp_latent:
                input_latent = torch.load(input_path + ".latent.pt")
            else:
                input_latent = torch.load(video + 'mean.latent.pt')
            target_latent = torch.load(target_path + ".latent.pt")
        else:
            target_latent = torch.tensor(0.)
            input_latent = torch.tensor(0.)

        return {
            'audio': audio,
            'target_img': target_img,
            'input_latent': input_latent,
            'target_latent': target_latent,
            'indices': indices,
            'paths': paths
        }


class StyleGANDataset(IterableDataset):
    def __init__(self, batch_size, downsample=True, device='cuda'):
        super(StyleGANDataset, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.downsample = downsample

        # Init generator
        self.g = Generator(1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        self.g.latent_avg = self.g.latent_avg.to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

    def __iter__(self):
        # Sample random z
        z = torch.randn(self.batch_size, 512, device=self.device)

        # Generate image
        with torch.no_grad():
            img, _ = self.g([z], truncation=0.9, truncation_latent=self.g.latent_avg)

        # Resize from 1024 to 256
        if self.downsample:
            img = downsample_256(img)

        yield {'x': img}


def get_video_paths_by_file(root_path, filename, max_frames_per_vid=-1):
    with open(filename, 'r') as f:
        lines = f.readlines()
    videos = [root_path + line.replace('\n', '') + '/' for line in lines]
    random.shuffle(videos)

    videos = [sorted([p.split('.')[0] for p in glob(v + '*.png')])[:max_frames_per_vid] for v in videos]

    return videos


class RandomAudioSampler(Sampler):
    """
    Samples batches of sequential indices of length T + 1 (last index is for
    random input frame).
    If weighted, the probability a video is chosen depends on its length.

    example usage:
        sampler = RandomTagesschauAudioSampler(paths, T=8, batch_size=args.batch_size)

    args:
        paths (list of lists):
        T (int):
        batch_size (int):
        num_batches (int):
        weighted (bool):
    """

    def __init__(self, paths, T, batch_size, num_batches, weighted=False, static_random=False):
        indices = []
        i = 0
        for path in paths:
            indices.append([])
            for p in path:
                indices[-1].append(i)
                i += 1
        self.indices = indices
        self.T = T
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.static_random = static_random
        if weighted:
            len_videos = [len(v) for v in indices]
            self.prob_video = [float(length) / sum(len_videos)
                               for length in len_videos]
        else:
            self.prob_video = [1. / len(self.indices) for _ in self.indices]

        # Use always the same random input for each video
        self.input_indices = [random.choice(ind) for ind in self.indices]

    def __iter__(self):
        batch = []
        videos = random.choices(self.indices, weights=self.prob_video, k=len(self))
        for video in videos:
            start = random.randint(0, len(video) - self.T)
            inp_idx = random.choice(video)
            sample = video[start: start + self.T] + [inp_idx]
            batch.append(sample)
        return iter(batch)
        batch = []
        video_inds = random.choices(range(len(self.indices)), weights=self.prob_video, k=len(self))
        for video_ind in video_inds:
            video = self.indices[video_ind]
            start = random.randint(0, len(video) - self.T)
            if self.static_random:
                inp_idx = self.input_indices[video_ind]
            else:
                inp_idx = random.choice(video)
            sample = video[start: start + self.T] + [inp_idx]
            batch.append(sample)
        return iter(batch)

    def __len__(self):
        return self.batch_size * self.num_batches
