"""
File to specify dataloaders for different datasets
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from PIL import Image
from torch.utils.data.dataset import Dataset

import utils


class RAVDESSDatasetVideo(Dataset):
    def __init__(self,
                 root_path):
        root_dir = pathlib.Path(root_path)

        # Get all paths
        all_paths = [str(p) for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        if len(all_paths) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root_path))

        # Get emotions
        emotions = [int(p.split('/')[-1][:-4].split('-')[2]) - 1
                    for p in all_paths]
        emotions = utils.np_int_to_one_hot(emotions, 8)

        self.all_paths = all_paths
        self.emotions = emotions

    def __getitem__(self, index):
        sequence = load_video(self.all_paths[index])

        return sequence, self.emotions[index]


def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_video(path):
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


ds = RAVDESSDatasetVideo('../Datasets/RAVDESS')

print(next(iter(ds)))


"""
Actor 1
mean (RGB): [212.11412638 202.15546712 199.68282704]
std (RGB): [76.99559435 86.44708823 89.27821175]
"""
