"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

import cv2
import numpy as np
import os
import pathlib

HOME = os.path.expanduser('~')


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
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(np.array(frame))

    all_frames = np.array(all_frames)
    print('Mean: {}'.format(all_frames.mean(axis=(0, 1, 2))))
    print('Std: {}'.format(all_frames.std(axis=(0, 1, 2))))


def ravdess_convert_png(root_path):
    target_path = HOME + '/Datasets/RAVDESS/Image'
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for folder in all_folders:
        paths = [str(p) for p in list(folder.glob('*/'))]
        actor = paths[0].split('/')[-2]
        os.makedirs(os.path.join(target_path, actor), exist_ok=True)
        for path in paths:
            file = path.split('/')[-1][:-4]
            i_frame = 0
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                i_frame += 1
                frame = np.array(frame)
                save_str = os.path.join(
                    target_path, actor, file + '-' + str(i_frame).zfill(3) + '.jpg')
                cv2.imwrite(save_str, frame)


# ravdess_get_mean(HOME + '/Datasets/RAVDESS/Video')
ravdess_convert_png(HOME + '/Datasets/RAVDESS/Video')
