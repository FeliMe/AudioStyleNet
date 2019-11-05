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
import random

from PIL import Image
from torchvision import transforms

from dataloader import RAVDESSDataset

HOME = os.path.expanduser('~')

IMAGE_224_PATH = HOME + '/Datasets/RAVDESS/Image224'
IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/Image128'
LANDMARKS_PATH = HOME + '/Datasets/RAVDESS/Landmarks'
VIDEO_PATH = HOME + '/Datasets/RAVDESS/Video'


def ravdess_get_mean_std_image(root_path, gray=False):
    root_dir = pathlib.Path(root_path)

    if gray:
        transform_list = [transforms.Grayscale(), transforms.ToTensor()]
    else:
        transform_list = [transforms.ToTensor()]

    transform = transforms.Compose(transform_list)

    # Get all paths
    all_sentences = [p for p in list(root_dir.glob('*/*/'))
                     if str(p).split('/')[-1] != '.DS_Store']
    num_sentences = len(all_sentences)

    # Use only 5 frames from each sentence
    all_frame_paths = []
    for path in all_sentences:
        sentence = list(path.glob('*'))
        random.shuffle(sentence)
        sentence = sentence[:5]
        sentence = [str(path) for path in sentence]
        for s in sentence:
            all_frame_paths.append(s)

    print("{} frames used from {} sentences".format(
        len(all_frame_paths), num_sentences))
    all_frames = []

    for frame_path in all_frame_paths:
        with open(frame_path, 'rb') as file:
            frame = Image.open(file).convert('RGB')
            frame = transform(frame)
            all_frames.append(frame.numpy())

    all_frames = np.array(all_frames)
    print(all_frames.shape)
    print('Mean: {}'.format(all_frames.mean(axis=(0, 2, 3))))
    print('Std: {}'.format(all_frames.std(axis=(0, 2, 3))))


def ravdess_convert_to_frames(root_path):
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    target_path = IMAGE_128_PATH
    target_size = 128
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for i_folder, folder in enumerate(all_folders):
        paths = [str(p) for p in list(folder.glob('*/'))]
        actor = paths[0].split('/')[-2]

        for i_path, path in enumerate(paths):
            utterance = path.split('/')[-1][:-4]
            path_to_utt = os.path.join(target_path, actor, utterance)
            print("Utterance {} of {}, actor {} of {}, {}".format(
                i_path + 1, len(paths), i_folder + 1, len(all_folders),
                path_to_utt))
            os.makedirs(path_to_utt, exist_ok=True)
            i_frame = 0
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                # Frame shape: (720, 1280, 3)
                ret, frame = cap.read()
                if not ret:
                    break
                i_frame += 1

                # Get target file name
                save_str = os.path.join(
                    path_to_utt, str(i_frame).zfill(3) + '.jpg')

                # If file already exists, skip
                if os.path.exists(save_str):
                    print("Already exists. Skipping...")
                    continue

                # Pre-resize to save computation (3 * target_size)
                shape = frame.shape
                w_ = 3 * target_size
                h_ = int((shape[1] / shape[0]) * w_)
                frame = cv2.resize(frame, (h_, w_))

                # Grayscale image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                rects = detector(frame, 1)
                for (i, rect) in enumerate(rects):
                    # Detect landmarks in faces
                    landmarks = predictor(gray, rect)
                    landmarks = shape_to_np(landmarks)

                    # Center of face
                    cx, cy = landmarks[33]

                    # Top and bottom
                    top = min(landmarks[19, 1], landmarks[24, 1])
                    bottom = max(landmarks[6:10, 1])
                    h = bottom - top

                    # Left and right
                    left = cx - h // 2
                    right = cx + h // 2

                    # Add margin
                    margin = int(.1 * h)
                    top -= int(1.8 * margin)
                    bottom += int(0.2 * margin)
                    left -= margin
                    right += margin

                # Visualize
                # cv2.rectangle(frame,
                #               (left, top),
                #               (right, bottom),
                #               (0, 0, 255), 1)
                # cv2.imshow("Output", frame)
                # cv2.waitKey(0)

                # Cut center
                frame = frame[top:bottom, left:right]

                # Resize
                frame = cv2.resize(frame, (target_size, target_size))

                # Save
                cv2.imwrite(save_str, frame)


def ravdess_extract_landmarks(root_path):
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    target_path = LANDMARKS_PATH
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


def ravdess_plot_label_distribution(data_path):
    ds = RAVDESSDataset(data_path)
    hist, _ = np.histogram(ds.emotions.numpy(), bins=8)
    hist = hist / len(ds.emotions)
    plt.bar(np.arange(8), hist)
    plt.title("Normalized distribution of RAVDESS dataset")
    plt.xticks(np.arange(8),
               ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])
    # plt.savefig('dist.jpg')
    plt.show()


# ravdess_get_mean_std_image(IMAGE_224_PATH, True)
ravdess_convert_to_frames(VIDEO_PATH)
# ravdess_extract_landmarks(IMAGE_224_PATH)
# ravdess_group_by_utterance(IMAGE_224_PATH)
# ravdess_plot_label_distribution(IMAGE_PATH)
