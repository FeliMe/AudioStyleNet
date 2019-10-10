"""
File to specify dataloaders for different datasets
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
                 format='image'):
        root_dir = pathlib.Path(root_path)

        # Get all paths
        all_paths = [str(p) for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        if len(all_paths) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root_path))

        random.shuffle(all_paths)
        if max_samples is not None:
            all_paths = all_paths[:min(len(all_paths), max_samples)]

        # Get emotions
        emotions = [int(p.split('/')[-1][:-4].split('-')[2]) - 1
                    for p in all_paths]
        emotions = torch.tensor(emotions, dtype=torch.long)

        self.all_paths = all_paths
        self.emotions = emotions
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7557917, 0.6731194, 0.65221864],
                                 std=[0.30093336, 0.3482375, 0.36186528]),
        ])

        if format == 'image':
            self.load_fn = load_image
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
        x = self.load_fn(self.all_paths[index], self.transforms)

        return x, self.emotions[index]

    def __len__(self):
        return len(self.all_paths)

    def show_sample(self):
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample)


def load_image(path, transform):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        img = transform(img)
        return img


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


def load_landmarks(path, transform):
    # Load
    landmarks = torch.tensor(np.load(path), dtype=torch.float)
    # Normalize
    landmarks = (landmarks - landmarks.mean(dim=0)) / landmarks.std(dim=0)
    return landmarks


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
    print(landmarks.shape)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()


"""
Mean (RGB) PIL: [0.7557917, 0.6731194, 0.65221864]
Std (RGB) PIL: [0.30093336, 0.3482375, 0.36186528]

mean (RGB) cv2: [212.11412638 202.15546712 199.68282704]
std (RGB) cv2: [76.99559435 86.44708823 89.27821175]
"""
