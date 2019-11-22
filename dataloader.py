"""
File to specify dataloaders for different datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import torch

from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import utils


class RAVDESSDataset(Dataset):
    """
    Dataset class for loading RAVDESS sentences in a sequential manner.

    Shortest sentence in RAVDESS has 94 frames.

    Output shapes:
        'image': [batch_size, sequence_length, 1 or 3, height, width]
        'landmarks': [batch_size, sequence_length, 68 * 2]

    Arguments:
        root_path (str): Path to data files
        data_format (str): Format of data files ('image' or 'landmarks')
        normalize (bool): Normalize data before outputting
        mean (list): Dataset mean values
        std (list): Dataset standard deviations
        max_samples (int or None): Maximum number of samples to be considered.
                                   Choose None for whole dataset
        seed (int): Random seed for reproducible shuffling
        sequence_length (int): Number of frames to be loaded per item
        step_size (int): Step size for loading a sequence
        image_size (int or tuple): Size of input images
    """
    def __init__(self,
                 root_path,
                 data_format='image',
                 normalize=True,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 max_samples=None,
                 seed=123,
                 sequence_length=1,
                 step_size=1,
                 image_size=64):

        assert (sequence_length * step_size) - 1 <= 94, \
            "Sequence is too long, step size too big or window size too" + \
            " big. Shortest sentence in RAVDESS is only 94 frames long."

        self.normalize = normalize
        self.mean = mean
        self.std = std

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        sentences = [str(p) for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        if len(sentences) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

        # Random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Shuffle sentences
        random.shuffle(sentences)

        if max_samples is not None:
            sentences = sentences[:min(len(sentences), max_samples)]

        trans = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.transforms = transforms.Compose(trans)

        self.sentences = sentences
        self.sequence_length = sequence_length
        self.step_size = step_size

        if data_format == 'image':
            self.load_fn = load_images
            self.show_fn = show_images
        elif data_format == 'landmarks':
            self.load_fn = load_landmarks
            self.show_fn = show_landmarks
        else:
            raise (RuntimeError('Unknown format {}'.format(data_format)))

    def _get_sample(self, sentence, indices):
        # Get paths to load
        paths = [os.path.join(sentence, str(idx).zfill(3))
                 for idx in indices]
        x = self.load_fn(paths, self.transforms)

        return x

    def _get_random_indices(self, sentence):
        len_sentence = len(list(pathlib.Path(sentence).glob('*')))
        rand_idx = torch.randint(1, len_sentence - self.sequence_length, (1,)).item()
        indices = list(range(rand_idx,
                             rand_idx + (self.sequence_length * self.step_size),
                             self.step_size))

        return indices

    def show_sample(self):
        """
        Plot a random sample
        """
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        # Select a sentence
        sentence = self.sentences[item]

        # Get sample
        x = self._get_sample(sentence, self._get_random_indices(sentence))

        # Get emotion
        emotion = int(sentence.split('/')[-1].split('-')[2]) - 1

        return {'x': x, 'y': emotion}


class RAVDESSDSPix2Pix(RAVDESSDataset):
    def __init__(self,
                 root_path,
                 target_root_path,
                 data_format='image',
                 use_same_sentence=True,
                 normalize=True,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 max_samples=None,
                 seed=123,
                 sequence_length=1,
                 step_size=1,
                 image_size=64):
        super(RAVDESSDSPix2Pix, self).__init__(root_path, data_format,
                                               normalize, mean, std,
                                               max_samples, seed,
                                               sequence_length, step_size,
                                               image_size)

        self.target_root_path = target_root_path
        self.use_same_sentence = use_same_sentence
        self.show_fn = show_pix2pix

        print(self.mean, self.std)

    def __getitem__(self, item):
        """
        Gets a pair of sequences (input b and target a). Source of the sequences
        is defined by root_path for input and target_root_path for target
        """

        # Input sentence
        input_sentence = self.sentences[item]
        indices = self._get_random_indices(input_sentence)
        a = self._get_sample(input_sentence, indices)

        # Target sentence
        if self.use_same_sentence:
            target_sentence = os.path.join(self.target_root_path, *input_sentence.split('/')[-2:])
        else:
            actor = os.path.join(self.target_root_path, *input_sentence.split('/')[-2:-1])
            all_sentences = [str(p) for p in list(pathlib.Path(actor).glob('*'))]
            target_sentence = random.choice(all_sentences)
            indices = self._get_random_indices(target_sentence)
        b = self._get_sample(target_sentence, indices)

        # Get emotion from target sentence
        emotion = int(target_sentence.split('/')[-1].split('-')[2]) - 1

        return {'A': a, 'B': b, 'y': emotion}

    def show_sample(self):
        """
        Plot a random sample
        """
        sample = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)


def load_images(paths, transform):
    x = []
    for path in paths:
        x.append(load_image(path + '.jpg', transform))
    return torch.stack(x, dim=0)


def load_image(path, transform):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        img = transform(img)
        return img


def load_landmarks(paths, transform):
    x = []
    for path in paths:
        x.append(load_landmark(path + '.npy', transform))
    return torch.cat(x, dim=0)


def load_landmark(path, transform):
    landmarks = torch.tensor(np.load(path), dtype=torch.float)
    return landmarks.reshape(-1)


def show_images(img, mean, std, normalize):
    """
    Plots a sequence of images
    """
    if normalize:
        transform = utils.denormalize(mean, std)
        img = torch.stack([transform(a) for a in img], 0)
    img = make_grid(img, nrow=img.size(0), normalize=True)
    plt.figure(figsize=(img.size(0), 1))
    plt.imshow(np.moveaxis(img.numpy(), 0, 2))
    plt.show()


def show_landmarks(landmarks, *kargs):
    if len(landmarks.shape) == 2:
        landmarks = landmarks[0]
    landmarks = landmarks[:2 * 68].reshape(-1, 2)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()


def show_pix2pix(sample, mean, std, normalize):
    """
    Plots a sample (input sequence and target sequence)
    """
    img_a = sample['A']
    img_b = sample['B']

    # Denormalize
    if normalize:
        transform = utils.denormalize(mean, std)
        img_a = torch.stack([transform(a) for a in img_a], 0)
        img_b = torch.stack([transform(b) for b in img_b], 0)

    # Make image grid
    imgs = torch.cat([img_a, img_b])
    imgs = make_grid(imgs, nrow=img_a.size(0), normalize=True)

    # Plot image
    plt.figure(figsize=(img_a.size(0), 2))
    plt.imshow(np.moveaxis(imgs.numpy(), 0, 2))
    plt.axis('off')
    plt.show()


def get_data_loaders(dataset, validation_split, batch_size, use_cuda):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = RandomSampler(train_indices)
    val_sampler = RandomSampler(val_indices)

    if use_cuda and torch.cuda.is_available():
        print("Pinning memory")
        kwargs = {'pin_memory': True}
    else:
        kwargs = {}

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=8,
                              sampler=train_sampler,
                              drop_last=True,
                              **kwargs)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            sampler=val_sampler,
                            drop_last=True,
                            **kwargs)

    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }

    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices)
    }

    return data_loaders, dataset_sizes