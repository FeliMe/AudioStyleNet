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
from tqdm import tqdm

from dataloader import RAVDESSDataset

HOME = os.path.expanduser('~')

IMAGE_224_PATH = HOME + '/Datasets/RAVDESS/Image224'
IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/Image128'
LANDMARKS_PATH = HOME + '/Datasets/RAVDESS/Landmarks'
LANDMARKS_128_PATH = HOME + '/Datasets/RAVDESS/Landmarks128'
LANDMARKS_POINT_IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/LandmarksPointImage128'
LANDMARKS_LINE_IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/LandmarksLineImage128'
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

    image_path = IMAGE_128_PATH
    landmarks_path = LANDMARKS_128_PATH
    target_size = 128
    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for i_folder, folder in enumerate(all_folders):
        paths = [str(p) for p in list(folder.glob('*/'))]
        actor = paths[0].split('/')[-2]

        for i_path, path in enumerate(tqdm(paths)):
            utterance = path.split('/')[-1][:-4]
            path_to_utt_img = os.path.join(image_path, actor, utterance)
            path_to_utt_landmarks = os.path.join(landmarks_path, actor, utterance)
            print("Utterance {} of {}, actor {} of {}, {}".format(
                i_path + 1, len(paths), i_folder + 1, len(all_folders),
                path_to_utt_img))
            os.makedirs(path_to_utt_img, exist_ok=True)
            os.makedirs(path_to_utt_landmarks, exist_ok=True)

            # h_ needs to be computed for every utterance
            h = None
            # Restart frame counter
            i_frame = 0

            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                # Frame shape: (720, 1280, 3)
                ret, frame = cap.read()
                if not ret:
                    break
                i_frame += 1

                # Get target file name
                save_str_img = os.path.join(
                    path_to_utt_img, str(i_frame).zfill(3) + '.jpg')
                save_str_landmarks = os.path.join(
                    path_to_utt_landmarks, str(i_frame).zfill(3) + '.jpg')

                # If file already exists, skip
                if os.path.exists(save_str_img):
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
                    top_ = min(landmarks[19, 1], landmarks[24, 1])
                    bottom_ = max(landmarks[6:10, 1])
                    left_ = min(landmarks[:4, 0])
                    right_ = max(landmarks[12:16, 0])
                    cx = (left_ + right_) // 2
                    cy = (top_ + bottom_) // 2

                    if h is None:
                        # Top and bottom
                        h = bottom_ - top_
                        # Add margin
                        margin = int(.35 * h)
                        h = h + margin

                # Compute left right
                if cx - (h // 2) < 0:
                    left = 0
                    right = h
                elif cx + (h // 2) > shape[1]:
                    right = shape[1]
                    left = shape[1] - h
                else:
                    left = cx - (h // 2)
                    right = cx + (h // 2)

                # Compute top bottom
                if cy - (h // 2) < 0:
                    top = 0
                    bottom = h
                elif cy + (h // 2) > shape[0]:
                    bottom = shape[0]
                    top = shape[0] - h
                else:
                    top = cy - (h // 2)
                    bottom = cy + (h // 2)

                # # Visualize
                # cv2.rectangle(frame,
                #               (left, top),
                #               (right, bottom),
                #               (0, 0, 255), 1)
                # for (x, y) in landmarks:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.imshow("Output", frame)
                # cv2.waitKey(0)

                # Cut center
                frame = frame[top:bottom, left:right]
                landmarks[:, 0] -= left
                landmarks[:, 1] -= top

                # Resize landmarks
                landmarks[:, 0] = landmarks[:, 0] * (target_size / frame.shape[1])
                landmarks[:, 1] = landmarks[:, 1] * (target_size / frame.shape[0])

                # Resize frame
                frame = cv2.resize(frame, (target_size, target_size))

                # # Visualize 2
                # for (x, y) in landmarks:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.imshow("Output", frame)
                # cv2.waitKey(0)

                # Save
                np.save(save_str_landmarks, landmarks)
                cv2.imwrite(save_str_img, frame)


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


def ravdess_landmark_to_point_image(data_path):

    target_path = LANDMARKS_POINT_IMAGE_128_PATH
    image_size = 128
    data_dir = pathlib.Path(data_path)

    all_files = [str(p) for p in list(data_dir.glob('*/*/*'))
                 if str(p).split('/')[-1] != '.DS_Store']

    for i_file, file in enumerate(tqdm(all_files)):
        save_dir = os.path.join(target_path, *file.split('/')[-3:-1])
        save_str = os.path.join(save_dir, file.split('/')[-1][:3] + '.jpg')

        # Load landmarks
        landmarks = np.load(file)

        # Create blank image
        img = np.zeros((image_size, image_size, 1), np.uint8)

        # Draw landmarks as circles
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 1, 255, -1)

        # Visualize
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)

        # Save image
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_str, img)


def ravdess_landmark_to_line_image(data_path):
    target_path = LANDMARKS_LINE_IMAGE_128_PATH
    image_size = 128
    data_dir = pathlib.Path(data_path)

    all_files = [str(p) for p in list(data_dir.glob('*/*/*'))
                 if str(p).split('/')[-1] != '.DS_Store']

    for i_file, file in enumerate(tqdm(all_files)):
        save_dir = os.path.join(target_path, *file.split('/')[-3:-1])
        save_str = os.path.join(save_dir, file.split('/')[-1][:3] + '.jpg')

        # Load landmarks
        landmarks = np.load(file)

        # Create blank image
        img = np.zeros((image_size, image_size, 1), np.uint8)

        # Draw face
        thickness = 1
        _draw_lines(img, landmarks[:17], thickness)  # Jaw line
        _draw_lines(img, landmarks[17:22], thickness)  # Right eyebrow
        _draw_lines(img, landmarks[22:27], thickness)  # Left eyebrow
        _draw_lines(img, landmarks[27:31], thickness)  # Nose vertical
        _draw_lines(img, landmarks[31:36], thickness)  # Nose horizontal
        cv2.drawContours(img, [landmarks[36:42]], 0, 255, thickness)  # Right eye
        cv2.drawContours(img, [landmarks[42:48]], 0, 255, thickness)  # Left eye
        cv2.drawContours(img, [landmarks[48:59]], 0, 255, thickness)  # Outer lips
        cv2.drawContours(img, [landmarks[60:]], 0, 255, thickness)  # Inner lips

        # Visualize
        cv2.imshow("Output", img)
        cv2.waitKey(0)

        # Save image
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(save_str, img)


def _draw_lines(img, points, thickness):
    for index, item in enumerate(points):
        if index == len(points) - 1:
            break
        cv2.line(img, tuple(item), tuple(points[index + 1]), 255, thickness)


# ravdess_get_mean_std_image(IMAGE_224_PATH, True)
# ravdess_extract_landmarks(IMAGE_224_PATH)
# ravdess_group_by_utterance(IMAGE_224_PATH)
# ravdess_plot_label_distribution(IMAGE_PATH)
# ravdess_convert_to_frames(VIDEO_PATH)
# ravdess_landmark_to_point_image(LANDMARKS_128_PATH)
ravdess_landmark_to_line_image(LANDMARKS_128_PATH)
