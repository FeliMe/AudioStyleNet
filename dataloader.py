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


IMG_NET_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMG_NET_STD = [0.229, 0.224, 0.225]  # ImageNet std


class RAVDESSDataset(Dataset):
    """
    Dataset class for loading RAVDESS sentences in a sequential manner.

    Shortest sentence in RAVDESS has 94 frames.

    Output shapes:

        if sequence_length > 1:
        'image': [batch_size, sequence_length, window_size * 3, height, width]
        'landmarks': [batch_size, sequence_length, window_size * 68 * 2]

        'image': [batch_size, window_size * 3, height, width]
        'landmarks': [batch_size, window_size * 68 * 2]

        if sequence_length == 1:

    Arguments:
        root_path (str): Path to data files
        data_format (str): Format of data files ('image' or 'landmarks')
        max_samples (int or None): Maximum number of samples to be considered.
                                   Choose None for whole dataset
        seed (int): Random seed for reproducible shuffling
        sequence_length (int): Number of frames to be loaded per item
        window_size (int): Number of frames after each target frame to be loaded
                           (including target frame)
        step_size (int): Step size for loading a sequence
    """
    def __init__(self,
                 root_path,
                 data_format='image',
                 use_gray=False,
                 max_samples=None,
                 seed=123,
                 sequence_length=1,
                 window_size=1,
                 step_size=1):

        assert (sequence_length * step_size) - 1 + (window_size - 1) <= 94, \
            "Sequence is too long, step size too big or window size too" + \
            " big. Shortest sentence in RAVDESS is only 94 frames long."

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        sentences = [p for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        if len(sentences) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

        # Shuffle sentences
        random.seed(seed)
        random.shuffle(sentences)

        if max_samples is not None:
            sentences = sentences[:min(len(sentences), max_samples)]

        # Count length of all sentences
        len_sentences = [len(list(sentence.glob('*')))
                         for sentence in sentences]

        # Convert sentence paths to strings
        sentences = [str(sentence) for sentence in sentences]

        # Get emotions
        emotions = [int(p.split('/')[-1].split('-')[2]) - 1
                    for p in sentences]
        emotions = torch.tensor(emotions, dtype=torch.long)

        # Add transforms
        trans = [transforms.ToTensor()]
        if use_gray:
            trans.insert(0, transforms.Grayscale())
        else:
            trans.append(transforms.Normalize(mean=IMG_NET_MEAN,
                                              std=IMG_NET_STD),)
        self.transforms = transforms.Compose(trans)

        self.emotions = emotions
        self.sentences = sentences
        self.len_sentences = len_sentences
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.step_size = step_size

        if data_format == 'image':
            self.load_fn = load_images
            self.show_fn = show_image
        elif data_format == 'landmarks':
            self.load_fn = load_landmarks
            self.show_fn = show_landmarks
        else:
            raise (RuntimeError('Unknown format {}'.format(data_format)))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        # Get paths to load
        rand_idx = np.random.randint(
            1, self.len_sentences[item] - self.sequence_length - (self.window_size - 1))

        indices = list(range(rand_idx,
                             rand_idx + (self.sequence_length * self.step_size),
                             self.step_size))

        if self.window_size > 1:
            indices = [list(range(i, i + self.window_size)) for i in indices]
            paths = [[os.path.join(self.sentences[item], str(i).zfill(3))
                      for i in idx] for idx in indices]
            x = []
            for path in paths:
                x.append(self.load_fn(path, self.transforms))

            if self.sequence_length == 1:
                x = torch.cat(x, dim=0)
            else:
                x = torch.stack(x, dim=0)
        else:
            paths = [os.path.join(self.sentences[item], str(idx).zfill(3))
                     for idx in indices]
            x = self.load_fn(paths, self.transforms)

        return x, self.emotions[item]

    def show_sample(self):
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample)


def load_images(paths, transform):
    x = []
    for path in paths:
        x.append(load_image(path + '.jpg', transform))
    return torch.cat(x, dim=0)


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


def load_video(path, transform):
    all_frames = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(np.array(frame))
    return np.array(all_frames)


def show_image(image):
    if len(image.shape) == 4:
        image = image[0]
    transform = utils.denormalize(IMG_NET_MEAN, IMG_NET_STD)
    image = transform(image[:3])
    plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    plt.show()


def show_landmarks(landmarks):
    if len(landmarks.shape) == 2:
        landmarks = landmarks[0]
    landmarks = landmarks[:2 * 68].reshape(-1, 2)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()


def compute_euclidean_landmark_feats(l):
    # From: Multi-Feature Based Emotion Recognition for Video Clips
    ul = (l[38] + l[37]) / 2
    dl = (l[41] + l[40]) / 2
    ur = (l[44] + l[43]) / 2
    dr = (l[46] + l[47]) / 2
    a = (dl + ul) / 2
    b = (dr + ur) / 2
    feats = torch.zeros(34)
    feats[0] = torch.dist(l[17], l[36])
    feats[1] = torch.dist(l[26], l[45])
    feats[2] = torch.dist(ul, dl)
    feats[3] = torch.dist(l[37], l[41])
    feats[4] = torch.dist(l[38], l[40])
    feats[5] = torch.dist(ur, dr)
    feats[6] = torch.dist(l[43], l[47])
    feats[7] = torch.dist(l[44], l[46])
    feats[8] = torch.dist(l[33], l[51])
    feats[9] = torch.dist(l[39], l[48])
    feats[10] = torch.dist(l[42], l[54])
    feats[11] = torch.dist(l[33], l[48])
    feats[12] = torch.dist(l[33], l[54])
    feats[13] = torch.dist(l[48], l[54])
    feats[14] = torch.dist(l[51], l[57])
    feats[15] = torch.dist(l[33], l[8])
    feats[16] = torch.dist(ul, a)
    feats[17] = torch.dist(dl, a)
    feats[18] = torch.dist(l[48], a)
    feats[19] = torch.dist(ur, b)
    feats[20] = torch.dist(dr, b)
    feats[21] = torch.dist(l[54], b)
    feats[22] = torch.dist(l[48], l[54]) / 2
    feats[23] = torch.dist(l[49], l[58])
    feats[24] = torch.dist(l[53], l[55])
    feats[25] = torch.dist(l[21], l[39])
    feats[26] = torch.dist(l[22], l[42])
    feats[27] = torch.dist(l[36], l[39]) / 2
    feats[28] = torch.dist(l[42], l[45]) / 2
    feats[29] = feats[27] + feats[28]
    feats[30] = torch.dist((l[17] + l[26]) / 2, l[33])
    feats[31] = torch.dist((l[21] + l[22]) / 2, l[33])
    feats[32] = torch.dist((l[48] + l[54]) / 2, l[33])
    feats[33] = torch.dist(l[62], l[66])
    return feats


"""
Mean (RGB) PIL: [0.7557917, 0.6731194, 0.65221864]
Std (RGB) PIL: [0.30093336, 0.3482375, 0.36186528]

mean (RGB) cv2: [212.11412638 202.15546712 199.68282704]
std (RGB) cv2: [76.99559435 86.44708823 89.27821175]
"""
