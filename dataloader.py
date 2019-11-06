"""
File to specify dataloaders for different datasets
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import utils


# ImageNet mean / std
IMG_NET_MEAN = [0.485, 0.456, 0.406]
IMG_NET_STD = [0.229, 0.224, 0.225]

# RAVDESS mean / std
RAVDESS_MEAN = [0.755, 0.673, 0.652]
RAVDESS_STD = [0.300, 0.348, 0.361]

# ImageNet mean / std (Grayscale)
RAVDESS_GRAY_MEAN = [0.694]
RAVDESS_GRAY_STD = [0.332]


class RAVDESSDataset(Dataset):
    """
    Dataset class for loading RAVDESS sentences in a sequential manner.

    Shortest sentence in RAVDESS has 94 frames.

    Output shapes:

        if sequence_length > 1:
            'image': [batch_size, sequence_length, 3, height, width]
            'landmarks': [batch_size, sequence_length, 68 * 2]

        if sequence_length == 1:
            'image': [batch_size, 3, height, width]
            'landmarks': [batch_size, 68 * 2]

    Arguments:
        root_path (str): Path to data files
        data_format (str): Format of data files ('image' or 'landmarks')
        max_samples (int or None): Maximum number of samples to be considered.
                                   Choose None for whole dataset
        seed (int): Random seed for reproducible shuffling
        sequence_length (int): Number of frames to be loaded per item
        step_size (int): Step size for loading a sequence
    """
    def __init__(self,
                 root_path,
                 data_format='image',
                 use_gray=False,
                 max_samples=None,
                 seed=123,
                 sequence_length=1,
                 step_size=1):

        assert (sequence_length * step_size) - 1 <= 94, \
            "Sequence is too long, step size too big or window size too" + \
            " big. Shortest sentence in RAVDESS is only 94 frames long."

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        sentences = [str(p) for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        if len(sentences) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

        # Shuffle sentences
        random.seed(seed)
        random.shuffle(sentences)

        if max_samples is not None:
            sentences = sentences[:min(len(sentences), max_samples)]

        # Add transforms
        if use_gray:
            self.mean = RAVDESS_GRAY_MEAN
            self.std = RAVDESS_GRAY_STD
            trans = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])
        else:
            self.mean = RAVDESS_MEAN
            self.std = RAVDESS_STD
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])
        self.transforms = trans

        self.sentences = sentences
        self.sequence_length = sequence_length
        self.step_size = step_size

        if data_format == 'image':
            self.load_fn = load_images
            self.show_fn = show_image
        elif data_format == 'landmarks':
            self.load_fn = load_landmarks
            self.show_fn = show_landmarks
        else:
            raise (RuntimeError('Unknown format {}'.format(data_format)))

    def _get_sample(self, sentence, len_sentence):
        # Get paths to load
        rand_idx = np.random.randint(1, len_sentence - self.sequence_length)

        indices = list(range(rand_idx,
                             rand_idx + (self.sequence_length * self.step_size),
                             self.step_size))

        paths = [os.path.join(sentence, str(idx).zfill(3))
                 for idx in indices]
        x = self.load_fn(paths, self.transforms)

        return x

    def show_sample(self):
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        # Get sequence
        sentence = self.sentences[item]
        x = self._get_sample(sentence,
                             len(list(pathlib.Path(sentence).glob('*'))))

        # Get emotion
        emotion = int(sentence.split('/')[-1].split('-')[2]) - 1

        return x, emotion


class RAVDESSDSPix2Pix(RAVDESSDataset):
    def __init__(self,
                 root_path,
                 data_format='image',
                 use_gray=False,
                 max_samples=None,
                 seed=123,
                 sequence_length=1,
                 step_size=1):
        super(RAVDESSDSPix2Pix, self).__init__(root_path, data_format,
                                               use_gray, max_samples, seed,
                                               sequence_length, step_size)

    def __getitem__(self, item):
        # Sequence A
        sentence_a = self.sentences[item]
        a = self._get_sample(sentence_a,
                             len(list(pathlib.Path(sentence_a).glob('*'))))

        # Sequence B
        actor = os.path.join('/', *sentence_a.split('/')[:-1])
        sentences = [str(p) for p in list(pathlib.Path(actor).glob('*/'))]
        sentence_b = random.choice(sentences)
        b = self._get_sample(sentence_b,
                             len(list(pathlib.Path(sentence_b).glob('*'))))

        return {'A': a, 'B': b}

    def show_sample(self):
        sample = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        if len(sample['A'].shape) == 4:
            # Sequence lenght > 1
            img_a = sample['A'][0]
            img_b = sample['B'][0]
        else:
            # Sequence length == 1
            img_a = sample['A']
            img_b = sample['B']

        # Denormalize
        transform = utils.denormalize(self.mean, self.std)
        img_a = transform(img_a)
        img_b = transform(img_b)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title("First frame of sequence A")
        plt.imshow(np.moveaxis(img_a.numpy(), 0, 2))

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title("First frame of sequence B")
        plt.imshow(np.moveaxis(img_b.numpy(), 0, 2))

        plt.tight_layout()
        plt.show()


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
    # landmarks = compute_euclidean_landmark_feats(landmarks)
    return landmarks.reshape(-1)


def show_image(image, mean, std):
    if len(image.shape) == 4:
        # Sequence length > 1
        image = image[0]
    transform = utils.denormalize(mean, std)
    image = transform(image)
    plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    plt.show()


def show_landmarks(landmarks, mean, std):
    if len(landmarks.shape) == 2:
        landmarks = landmarks[0]
    landmarks = landmarks[:2 * 68].reshape(-1, 2)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()
