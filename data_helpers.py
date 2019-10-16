"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

import cv2
import dlib
import math
import numpy as np
import os
import pathlib
import random

from PIL import Image
from torchvision import transforms

HOME = os.path.expanduser('~')


def ravdess_get_mean_std_image(root_path):
    root_dir = pathlib.Path(root_path)

    transform = transforms.ToTensor()

    # Get all paths
    all_folders = [p for p in list(root_dir.glob('*/*/'))
                   if str(p).split('/')[-1] != '.DS_Store']
    num_actors = len([p for p in list(root_dir.glob('*/'))])

    # Use only 1000 frames from each actor
    all_actors = []
    for path in all_folders:
        actor = list(path.glob('*'))
        random.shuffle(actor)
        actor = actor[:1000]
        actor = [str(path) for path in actor]
        for a in actor:
            all_actors.append(a)

    print("{} frames used from {} actors".format(len(all_actors), num_actors))
    all_frames = []

    for actor in all_actors:
        with open(actor, 'rb') as f:
            frame = Image.open(f).convert('RGB')
            frame = transform(frame)
            all_frames.append(frame.numpy())

    all_frames = np.array(all_frames)
    print(all_frames.shape)
    print('Mean: {}'.format(all_frames.mean(axis=(0, 2, 3))))
    print('Std: {}'.format(all_frames.std(axis=(0, 2, 3))))


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
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    target_path = HOME + '/Datasets/RAVDESS/Landmarks'
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for i_folder, folder in enumerate(all_folders):
        actor = str(folder).split('/')[-1]
        os.makedirs(os.path.join(target_path, actor), exist_ok=True)
        paths = [str(p) for p in list(folder.glob('*/'))
                 if str(p).split('/')[-1] != '.DS_Store']

        for i_path, path in enumerate(paths):
            print("Detecting from image {} of {}, folder {} of {}".format(
                i_path, len(paths), i_folder, len(all_folders)))
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
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
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
    # Source https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def ravdess_group_by_utterance(root_path):
    root_dir = pathlib.Path(root_path)

    print(root_dir)

    # Get all paths
    all_actors = [p for p in list(root_dir.glob('*/'))
                  if str(p).split('/')[-1] != '.DS_Store']

    for actor in all_actors:
        print("Processing {}".format(str(actor).split('/')[-1]))
        frames = [str(frame) for frame in list(actor.glob('*'))]
        for i_frame, frame in enumerate(frames):
            new_folder = frame[:-8]
            new_path = os.path.join(new_folder, frame[-7:])
            os.makedirs(new_folder, exist_ok=True)
            os.rename(frame, new_path)


# ravdess_get_mean_std_image(HOME + '/Datasets/RAVDESS/Image')
# ravdess_convert_jpg(HOME + '/Datasets/RAVDESS/Video')
# ravdess_extract_landmarks(HOME + '/Datasets/RAVDESS/Image')
# ravdess_group_by_utterance(HOME + '/Datasets/RAVDESS/Image')
