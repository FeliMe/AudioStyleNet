"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

import cv2
import numpy as np
import pathlib


def ravdess_get_mean(root_path):
    root_dir = pathlib.Path(root_path)

    # Get all paths
    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    all_actors = [str(list(path.glob('*'))[0]) for path in all_folders]

    all_frames = []

    for actor in all_actors:
        print("Reading {}".format(actor.split('/')[-2]))
        cap = cv2.VideoCapture(actor)
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(np.array(frame))

    all_frames = np.array(all_frames)
    print('Mean: {}'.format(all_frames.mean(axis=(0, 1, 2))))
    print('Std: {}'.format(all_frames.std(axis=(0, 1, 2))))


ravdess_get_mean('../Datasets/RAVDESS')
