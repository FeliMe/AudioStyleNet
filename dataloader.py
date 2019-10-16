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


class RAVDESSDataset(Dataset):
    def __init__(self,
                 root_path,
                 max_samples=None,
                 sequence_length=1,
                 format='image'):
        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        all_sentences = [p for p in list(root_dir.glob('*/*'))
                         if str(p).split('/')[-1] != '.DS_Store']

        if len(all_sentences) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

        random.shuffle(all_sentences)
        if max_samples is not None:
            all_sentences = all_sentences[:min(len(all_sentences), max_samples)]

        # Count length of all sentences
        len_sentences = [len(list(sentence.glob('*')))
                         for sentence in all_sentences]

        # Convert sentence paths to strings
        all_sentences = [str(sentence) for sentence in all_sentences]

        # Get emotions
        emotions = [int(p.split('/')[-1].split('-')[2]) - 1
                    for p in all_sentences]
        emotions = torch.tensor(emotions, dtype=torch.long)

        self.all_sentences = all_sentences
        self.len_sentences = len_sentences
        self.sequence_length = sequence_length
        self.emotions = emotions
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7557917, 0.6731194, 0.65221864],
                                 std=[0.30093336, 0.3482375, 0.36186528]),
        ])

        if format == 'image':
            self.load_fn = load_images
            self.show_fn = show_image
        elif format == 'video':
            self.load_fn = load_video
            self.load_fn = show_image
        elif format == 'landmarks':
            self.load_fn = load_landmarks
            self.show_fn = show_landmarks
        else:
            raise (RuntimeError('Unknown format {}'.format(format)))

    def __getitem__(self, index):
        # Get paths to load
        rand_idx = np.random.randint(
            1, self.len_sentences[index] - self.sequence_length)
        indices = list(range(rand_idx, rand_idx + self.sequence_length))
        paths = [os.path.join(self.all_sentences[index], str(idx).zfill(3))
                 for idx in indices]

        # Load files
        x = self.load_fn(paths, self.transforms)

        return x, self.emotions[index]

    def __len__(self):
        return len(self.all_sentences)

    def show_sample(self):
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample[:, 0])

    def plot_label_distribution(self):
        hist, _ = np.histogram(self.emotions.numpy(), bins=8)
        hist = hist / len(self.emotions)
        plt.bar(np.arange(8), hist)
        plt.title("Normalized distribution of RAVDESS dataset")
        plt.xticks(np.arange(8),
                   ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])
        plt.savefig('dist.jpg')


def load_images(paths, transform):
    x = []
    for path in paths:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path + '.jpg', 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = transform(img)
            x.append(img)
    return torch.cat(x, dim=0)


def load_landmarks(paths, transform):
    x = []
    for path in paths:
        # Load
        landmarks = torch.tensor(np.load(path + '.npy'), dtype=torch.float)
        # Normalize
        landmarks = (landmarks - landmarks.mean(dim=0)) / landmarks.std(dim=0)
        # Reshape and append
        x.append(landmarks.reshape(-1))
    return torch.stack(x, dim=1)


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
        plt.imshow(np.moveaxis(image.numpy()[0], 0, 2))
    else:
        transform = utils.denormalize([0.7557917, 0.6731194, 0.65221864],
                                      [0.30093336, 0.3482375, 0.36186528])
        image = transform(image)
        plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    plt.show()


def show_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 2)
    print(landmarks.shape)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()


"""
Mean (RGB) PIL: [0.7557917, 0.6731194, 0.65221864]
Std (RGB) PIL: [0.30093336, 0.3482375, 0.36186528]

mean (RGB) cv2: [212.11412638 202.15546712 199.68282704]
std (RGB) cv2: [76.99559435 86.44708823 89.27821175]
"""
