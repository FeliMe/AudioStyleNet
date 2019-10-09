"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

import cv2
import dlib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

from collections import OrderedDict
from PIL import Image
from torchvision import transforms

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


def ravdess_convert_jpg(root_path):
    target_path = HOME + '/Datasets/RAVDESS/Image'
    target_size = 224
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for i_folder, folder in enumerate(all_folders):
        paths = [str(p) for p in list(folder.glob('*/'))]
        actor = paths[0].split('/')[-2]
        os.makedirs(os.path.join(target_path, actor), exist_ok=True)

        for i_path, path in enumerate(paths):
            print("File {} of {}, actor {} of {}".format(
                i_path, len(paths), i_folder, len(all_folders)))
            file = path.split('/')[-1][:-4]
            i_frame = 0
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                i_frame += 1

                # Resize
                height, width, _ = frame.shape
                new_height = target_size
                new_width = math.floor((target_size / height) * width)
                frame = cv2.resize(frame, (new_width, new_height))

                # Center crop
                left = int((new_width - target_size) / 2)
                frame = frame[:, left:left + target_size]

                # Save
                save_str = os.path.join(
                    target_path, actor, file + '-' + str(i_frame).zfill(3) + '.jpg')
                cv2.imwrite(save_str, frame)


def ravdess_extract_landmarks(root_path):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    target_path = HOME + '/Datasets/RAVDESS/Landmarks'
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for folder in all_folders:
        print("Starting folder {}".format(folder))
        actor = str(folder).split('/')[-1]
        os.makedirs(os.path.join(target_path, actor), exist_ok=True)
        paths = [str(p) for p in list(folder.glob('*/'))
                 if str(p).split('/')[-1] != '.DS_Store']

        for i_path, path in enumerate(paths):
            print("Detecting from image {} of {}".format(i_path, len(paths)))
            # load the input image, resize it, and convert it to grayscale
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            save_path = os.path.join(
                target_path, actor, path.split('/')[-1][:-4] + '.npy')

            # Detect faces
            rects = detector(img, 1)
            for (i, rect) in enumerate(rects):
                # Detect landmarks in faces
                landmarks = predictor(gray, rect)
                landmarks = shape_to_np(landmarks)

                # for (x, y) in landmarks:
                #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                # cv2.imshow("Output", img)
                # cv2.waitKey(0)

                # Save
                np.save(save_path, landmarks)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(landmarks, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# ravdess_get_mean(HOME + '/Datasets/RAVDESS/Video')
# ravdess_convert_jpg(HOME + '/Datasets/RAVDESS/Video')
ravdess_extract_landmarks(HOME + '/Datasets/RAVDESS/Image')
