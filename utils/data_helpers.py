"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

"""
Download files from google drive

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=CODE_FROM_ABOVE&id=FILEID'

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pQy1YUGtHeUM2dUE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=&id=0B7EVK8r0v71pQy1YUGtHeUM2dUE'
0B7EVK8r0v71pQy1YUGtHeUM2dUE
"""

import bz2
import cv2
import dlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import subprocess
import sys
import time
import torch

from azure.cognitiveservices.vision.face import FaceClient
from dreiDDFA.model import dreiDDFA
from glob import glob
from msrest.authentication import CognitiveServicesCredentials
from multiprocessing import Process
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage import io
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from utils import get_mouth_params, VideoAligner

# from utils.dataloader import RAVDESSDataset

HOME = os.path.expanduser('~')

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

IMAGE_256_PATH = HOME + '/Datasets/RAVDESS/Image256'
IMAGE_256_CROP_PATH = HOME + '/Datasets/RAVDESS/Image256Crop'
IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/Image128'
LANDMARKS_PATH = HOME + '/Datasets/RAVDESS/Landmarks'
LANDMARKS_128_PATH = HOME + '/Datasets/RAVDESS/Landmarks128'
LANDMARKS_256_PATH = HOME + '/Datasets/RAVDESS/Landmarks256'
LANDMARKS_POINT_IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/LandmarksPointImage128'
LANDMARKS_LINE_IMAGE_128_PATH = HOME + '/Datasets/RAVDESS/LandmarksLineImage128'
VIDEO_PATH = HOME + '/Datasets/RAVDESS/Video'

CELEBA_PATH = HOME + '/Datasets/CELEBA/Imgs'
CELEBA_LANDMARKS_PATH = HOME + '/Datasets/CELEBA/Landmarks'
CELEBA_LANDMARKS_LINE_IMAGE_PATH = HOME + '/Datasets/CELEBA/LandmarksLineImage'


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


def ravdess_to_frames_center_crop(root_path):
    image_path = IMAGE_256_CROP_PATH
    target_size = 256

    root_dir = pathlib.Path(root_path)

    all_folders = [p for p in list(root_dir.glob('*/'))
                   if str(p).split('/')[-1] != '.DS_Store']

    for i_folder, folder in enumerate(all_folders):
        paths = [str(p) for p in list(folder.glob('*/'))]
        actor = paths[0].split('/')[-2]

        for i_path, path in enumerate(tqdm(paths)):
            utterance = path.split('/')[-1][:-4]
            path_to_utt_img = os.path.join(image_path, actor, utterance)
            print("Utterance {} of {}, actor {} of {}, {}".format(
                i_path + 1, len(paths), i_folder + 1, len(all_folders),
                path_to_utt_img))
            os.makedirs(path_to_utt_img, exist_ok=True)

            # Restart frame counter
            i_frame = 0

            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                # Frame shape: (720, 1280, 3)
                ret, frame = cap.read()
                if not ret:
                    break
                i_frame += 1

                save_str_img = os.path.join(
                    path_to_utt_img, str(i_frame).zfill(3) + '.jpg')

                h, w, c = frame.shape
                new_w = int((w / h) * target_size)
                frame = cv2.resize(frame, (new_w, target_size))

                # print(frame.shape)

                # # Center crop
                # left = (new_w - target_size) // 2
                # right = left + target_size
                # frame = frame[:, left:right]

                # print(frame.shape)

                # Visualize
                cv2.imshow("Output", frame)
                cv2.waitKey(0)

                # Save
                # cv2.imwrite(save_str_img, frame)


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def align_image(
        frame,
        landmarks,
        output_size=256,
        transform_size=1024,
        enable_padding=True):
    """
    Source: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    """

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(landmarks)
    # lm_chin = lm[0: 17]  # left-right
    # lm_eyebrow_left = lm[17: 22]  # left-right
    # lm_eyebrow_right = lm[22: 27]  # left-right
    # lm_nose = lm[27: 31]  # top-down
    # lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    # lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Convert image to PIL
    img = Image.fromarray(frame)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(
            float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
        np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
        np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] -
                                                                   img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(
            img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(
            w - 1 - x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * \
            np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(
            np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size),
                        Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img


def ravdess_align_videos(root_path, group):
    if root_path[-1] != '/':
        root_path += '/'

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')

    actors = sorted(glob(root_path + '*/'))
    assert len(actors) > 0

    groups = []
    n = len(actors) // 7
    for i in range(0, len(actors), n):
        groups.append(actors[i:i + n])

    actors = groups[group]
    print(f"Group {group}, num_videos {len(actors)}, {len(groups)} groups in total")

    for actor in actors:
        sentences = sorted(glob(actor + '*.mp4'))
        target_actor = os.path.join(target_path, actor.split('/')[-2])

        for i_path, path in enumerate(tqdm(sentences)):
            utterance = path.split('/')[-1][:-4]
            save_dir = os.path.join(target_actor, utterance) + '/'
            print("Utterance {} of {}, {}".format(
                i_path + 1, len(sentences), save_dir))
            os.makedirs(save_dir, exist_ok=True)

            aligner = VideoAligner()
            aligner.align_video(path, save_dir)


def ravdess_gather_info(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    save_path = root_path + 'latent_data.pt'
    print(f"Saving to {save_path}")

    actors = sorted(glob(root_path + '*/'))
    assert len(actors) > 0

    names = []
    latents = []
    landmarks = []
    lm3d = []
    logits_fer = []
    logits_rav = []
    emotions = []
    for actor in tqdm(actors):
        sentences = sorted(glob(actor + '*/'))
        for sentence in sentences:
            frames = sorted(glob(sentence + '*.png'))
            for frame in frames:
                base = frame.split('.')[0]

                names.append('/'.join(base.split('/')[-2:]))
                emotions.append(int(base.split('/')[-2].split('-')[2]) - 1)
                latents.append(torch.load(base + '.latent.pt'))
                landmarks.append(torch.load(base + '.landmarks.pt'))
                lm3d.append(torch.load(base + '.landmarks3d.pt'))
                logits_fer.append(torch.load(base + '-logit_fer.pt'))
                logits_rav.append(torch.load(base + '-logit_ravdess.pt'))

    names = np.array(names)
    latents = torch.stack(latents)
    landmarks = torch.stack(landmarks)
    lm3d = torch.stack(lm3d)
    logits_fer = torch.stack(logits_fer)
    logits_rav = torch.stack(logits_rav)
    emotions = torch.tensor(emotions)

    data = {
        'names': names,
        'latents': latents,
        'landmarks': landmarks,
        'lm3d': lm3d,
        'logits_fer': logits_fer,
        'logits_rav': logits_rav,
        'emotions': emotions
    }

    torch.save(data, save_path)


def ravdess_encode_frames(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    actors = sorted(glob(root_path + '*/'))
    assert len(actors) > 0

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoder
    from my_models.models import resnetEncoder
    e = resnetEncoder(net=18).eval()
    state_dict = torch.load('saves/pre-trained/resNet18RAVDESS.pt')
    e.load_state_dict(state_dict)
    e = e.to(device)

    # Get latent avg
    from my_models.style_gan_2 import Generator
    g = Generator(1024, 512, 8, pretrained=True).eval()
    g.noises = [n.to(device) for n in g.noises]
    latent_avg = g.latent_avg.view(1, -1).repeat(18, 1)

    # transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for actor in tqdm(actors):
        sentences = sorted(glob(actor + '*/'))
        for sentence in sentences:
            frames = sorted(glob(sentence + '*.png'))
            for frame in frames:
                save_path = frame.split('.')[0] + '.latent.pt'
                # print(save_path)
                if os.path.exists(save_path):
                    continue

                # Load image
                img = t(Image.open(frame)).unsqueeze(0).to(device)

                # Encoder image
                with torch.no_grad():
                    latent_offset = e(img)[0].cpu()
                    latent = latent_offset + latent_avg

                # Visualize
                from torchvision.utils import make_grid
                from utils.utils import downsample_256
                print(save_path, latent.shape)
                img_gen = g.to(device)(
                    [latent.to(device)], input_is_latent=True, noise=g.noises)[0].cpu()
                img_gen = downsample_256(img_gen)
                img_gen = make_grid(
                    torch.cat((img_gen, img.cpu()), dim=0), normalize=True, range=(-1, 1))
                img_gen = transforms.ToPILImage('RGB')(img_gen)
                img_gen.show()
                1 / 0

                # Save
                # torch.save(latent, save_path)


def ravdess_resize_frames(path_to_actor):
    def downsample_img(img):
        c, h, w = img.shape
        factor = h // 256
        img = img.reshape(c, h // factor, factor, w // factor, factor)
        img = img.mean([2, 4])
        return img

    transform = transforms.ToTensor()
    if path_to_actor[-1] == '/':
        path_to_actor = path_to_actor[:-1]
    new_base_dir = os.path.join('/', *path_to_actor.split('/')[:-2], 'Aligned_256')
    os.makedirs(new_base_dir, exist_ok=True)
    new_dir = os.path.join(new_base_dir, path_to_actor.split('/')[-1])
    os.makedirs(new_dir, exist_ok=True)
    print('Saving to: {}'.format(new_dir))

    all_folders = [str(f) for f in list(pathlib.Path(path_to_actor).glob('*'))]
    all_folders = sorted(all_folders)

    for folder in tqdm(all_folders):
        save_dir = os.path.join(new_dir, folder.split('/')[-1])
        os.makedirs(save_dir, exist_ok=True)
        all_frames = [str(f) for f in pathlib.Path(folder).glob('*')]
        for frame in all_frames:
            save_path = os.path.join(save_dir, frame.split('/')[-1])
            image = transform(Image.open(frame))
            image = downsample_img(image)
            save_image(image, save_path)


def ravdess_convert_to_frames(root_path):
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    image_path = IMAGE_256_PATH
    landmarks_path = LANDMARKS_256_PATH
    target_size = 256
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

                # Pre-resize to save computation (1.65 * target_size)
                shape = frame.shape
                w_ = int(1.65 * target_size)  # new ds
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
                        # margin = int(.35 * h)  # Old ds
                        margin = int(.85 * h)  # new ds
                        h = h + margin

                    # shift cy
                    cy -= int(.15 * h)  # new ds

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


def ravdess_extract_landmarks(path_to_actor):
    # Source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/shape_predictor_68_face_landmarks.dat')

    if path_to_actor[-1] == '/':
        path_to_actor = path_to_actor[:-1]
    new_dir_lm = os.path.join('/', *path_to_actor.split('/')[:-2],
                              'Landmarks_Aligned256', path_to_actor.split('/')[-1])
    new_dir_mask = os.path.join('/', *path_to_actor.split('/')[:-2],
                                'Mask_Aligned256', path_to_actor.split('/')[-1])
    os.makedirs(new_dir_lm, exist_ok=True)
    os.makedirs(new_dir_mask, exist_ok=True)
    print('Saving to {} and {}'.format(new_dir_lm, new_dir_mask))

    all_folders = [str(f) for f in list(pathlib.Path(path_to_actor).glob('*'))]
    all_folders = sorted(all_folders)

    for folder in tqdm(all_folders):
        save_dir_lm = os.path.join(new_dir_lm, folder.split('/')[-1])
        save_dir_mask = os.path.join(new_dir_mask, folder.split('/')[-1])
        os.makedirs(save_dir_lm, exist_ok=True)
        os.makedirs(save_dir_mask, exist_ok=True)
        all_frames = [str(f) for f in pathlib.Path(folder).glob('*')
                      if str(f).split('/')[-1] != '.DS_Store']
        for frame in all_frames:
            # load the input image, resize it, and convert it to grayscale
            img = cv2.imread(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            save_path_lm = os.path.join(save_dir_lm, frame.split('/')[-1][:-4] + '.npy')
            save_path_mask = os.path.join(save_dir_mask, frame.split('/')[-1][:-4] + '.png')

            # Detect faces
            rects = detector(img, 1)
            for (i, rect) in enumerate(rects):
                # Detect landmarks in faces
                landmarks = predictor(gray, rect)
                landmarks = shape_to_np(landmarks)

                # Compute mask
                mask = compute_face_mask(landmarks, img)

                # Save
                np.save(save_path_lm, landmarks)

                # Save image
                cv2.imwrite(save_path_mask, mask)


def ravdess_get_landmarks(root, group):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat')

    frames = sorted(glob(root + '*/*/*.png'))

    groups = []
    n = len(frames) // 7
    for i in range(0, len(frames), n):
        groups.append(frames[i:i + n])

    frames = groups[group]
    print(
        f"Group {group}, num_frames {len(frames)}, {len(groups)} groups in total")

    for frame in tqdm(frames):
        save_path = frame.split('.')[0] + '.landmarks.pt'
        # if os.path.exists(save_path):
        #     continue
        img = io.imread(frame)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces
        rects = detector(img, 1)
        for (i, rect) in enumerate(rects):
            # Detect landmarks in faces
            landmarks = predictor(gray, rect)
            landmarks = torch.tensor(shape_to_np(landmarks))

            # Visualize
            # print(save_path)
            # for (x, y) in landmarks:
            #     cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
            # cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # 1 / 0

            # Save
            torch.save(landmarks, save_path)

            break


def compute_face_mask(landmarks, image):
    jaw = landmarks[0:17]
    left_eyebrow = landmarks[17:20]
    left_eyebrow[:, 1] = left_eyebrow[:, 1] - 10
    right_eyebrow = landmarks[24:27]
    right_eyebrow[:, 1] = right_eyebrow[:, 1] - 10
    hull = np.concatenate(
        (jaw, np.flip(right_eyebrow, 0), np.flip(left_eyebrow, 0)))
    mask = np.zeros(image.shape, dtype='uint8')
    mask = cv2.drawContours(mask, [hull], -1,
                            (255, 255, 255), thickness=cv2.FILLED)
    mask = cv2.bitwise_not(mask)
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
    return mask


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
        img = _draw_face(img, landmarks, 1)

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


def _draw_face(img, landmarks, thickness):
    _draw_lines(img, landmarks[:17], thickness)  # Jaw line
    _draw_lines(img, landmarks[17:22], thickness)  # Right eyebrow
    _draw_lines(img, landmarks[22:27], thickness)  # Left eyebrow
    _draw_lines(img, landmarks[27:31], thickness)  # Nose vertical
    _draw_lines(img, landmarks[31:36], thickness)  # Nose horizontal
    cv2.drawContours(img, [landmarks[36:42]], 0, 255, thickness)  # Right eye
    cv2.drawContours(img, [landmarks[42:48]], 0, 255, thickness)  # Left eye
    cv2.drawContours(img, [landmarks[48:59]], 0, 255, thickness)  # Outer lips
    cv2.drawContours(img, [landmarks[60:]], 0, 255, thickness)  # Inner lips
    return img


def celeba_extract_landmarks(root_path, target_path, line_img_path):
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(line_img_path, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    root_dir = pathlib.Path(root_path)

    all_files = [str(p) for p in list(root_dir.glob('*'))
                 if str(p).split('/')[-1] != '.DS_Store']

    for i_file, file in enumerate(tqdm(all_files)):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        save_path = os.path.join(
            target_path, file.split('/')[-1][:-4] + '.npy')
        save_line_img_path = os.path.join(
            line_img_path, file.split('/')[-1][:-4] + '.jpg')

        # Detect faces
        rects = detector(img, 1)
        for (i, rect) in enumerate(rects):
            # Detect landmarks in faces
            landmarks = predictor(gray, rect)
            landmarks = shape_to_np(landmarks)
            # Save
            np.save(save_path, landmarks)

            # Visualize
            # for (x, y) in landmarks:
            #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            # cv2.imshow("Output", img)
            # cv2.waitKey(0)

            # Create line img
            line_img = np.zeros_like(gray)
            # Draw face
            line_img = _draw_face(line_img, landmarks, 1)
            # Save image
            cv2.imwrite(save_line_img_path, line_img)

            # Visualize
            # cv2.imshow("Output", line_img)
            # cv2.waitKey(0)


def ravdess_project_to_latent(path_to_actor):
    from my_models.style_gan_2 import Generator
    from projector import Projector

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    g = Generator(1024, 512, 8, pretrained=True).to(device).train()
    for param in g.parameters():
        param.requires_grad = False

    proj = Projector(g)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if path_to_actor[-1] == '/':
        path_to_actor = path_to_actor[:-1]
    new_dir = os.path.join('/', *path_to_actor.split('/')[:-2],
                           'Projected', path_to_actor.split('/')[-1])
    os.makedirs(new_dir, exist_ok=True)
    print('Saving to {}'.format(new_dir))

    sentences = [str(f) for f in list(pathlib.Path(path_to_actor).glob('*'))]
    sentences = sorted(sentences)

    mapping = {
        'neutral': '01',
        'calm': '02',
        'happy': '03',
        'sad': '04',
        'angry': '05',
        'fearful': '06',
        'disgust': '07',
        'surprised': '08'
    }
    emotions = [mapping[e] for e in ['neutral', 'calm', 'fearful', 'disgust', 'surprised']]
    sentences = list(filter(lambda s: s.split('/')[-1].split('-')[2]
                            in emotions, sentences))
    print(sentences)
    1 / 0

    for folder in tqdm(sentences):
        save_dir = os.path.join(new_dir, folder.split('/')[-1])
        os.makedirs(save_dir, exist_ok=True)
        all_frames = [str(f) for f in pathlib.Path(folder).glob('*')
                      if str(f).split('/')[-1] != '.DS_Store']
        for i, frame in enumerate(sorted(all_frames)):
            print('Projecting {}'.format(frame))

            save_path = os.path.join(save_dir, frame.split('/')[-1][:-4] + '.pt')

            target_image = Image.open(frame)
            target_image = transform(target_image).to(device)

            # Run projector
            proj.run(target_image, 1000 if i == 0 else 50)

            # Collect results
            latents = proj.get_latents().cpu()
            torch.save(latents, save_path)


def ravdess_get_scores(root_path, model='fer'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model == 'fer':
        from my_models.models import FERClassifier
        model = FERClassifier(softmaxed=False).eval().to(device)
        appendix = '-logit_fer.pt'
    elif model == 'ravdess':
        from my_models.models import EmotionClassifier
        model = EmotionClassifier(softmaxed=False).eval().to(device)
        appendix = '-logit_ravdess.pt'
    else:
        raise NotImplementedError

    if root_path[-1] == '/':
        root_path = root_path[:-1]

    actors = sorted([str(p) for p in list(pathlib.Path(root_path).glob('*/'))
                     if str(p).split('/')[-1] != '.DS_Store'])

    t = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for actor in actors:
        print(actor.split('/')[-1])
        sentences = [str(s) for s in pathlib.Path(actor).glob('*/')
                     if str(s).split('/')[-1] != '.DS_Store']

        for folder in tqdm(sentences):
            frames = sorted([str(f)
                             for f in pathlib.Path(folder).glob('*.png')])
            for i, frame in enumerate(sorted(frames)):
                save_path = frame.split('.')[0] + appendix

                img = t(Image.open(frame)).unsqueeze(0).to(device)
                logits = model(img)[0].cpu()
                # print(save_path)
                # print(logits)
                # 1 / 0

                torch.save(logits, save_path)


def ravdess_azure_scores(actor):

    if actor[-1] == '/':
        actor = actor[:-1]

    KEY = 'b33dd9ddc4134928bb0af7e58aad8545'
    ENDPOINT = 'https://testfacefelime.cognitiveservices.azure.com/'

    # Create an authenticated FaceClient.
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    sentences = [str(s) for s in pathlib.Path(actor).glob('*/')
                 if str(s).split('/')[-1] != '.DS_Store']

    counter = 0

    for folder in tqdm(sentences):
        frames = sorted([str(f) for f in pathlib.Path(folder).glob('*.png')])
        for i, frame in enumerate(sorted(frames)):
            save_path = frame.split('.')[0] + '-score_azure.pt'

            detected_faces = face_client.face.detect_with_stream(
                image=open(frame, 'r+b'),
                return_face_id=False,
                return_face_landmarks=False,
                return_face_attributes=['emotion'],
                recognition_model='recognition_01'
            )
            counter += 1

            if not detected_faces:
                continue

            for face in detected_faces:
                emotions = face.face_attributes.emotion
                scores = torch.tensor([
                    emotions.neutral,
                    emotions.contempt,
                    emotions.happiness,
                    emotions.sadness,
                    emotions.anger,
                    emotions.fear,
                    emotions.disgust,
                    emotions.surprise
                ])
                break
            # print(save_path)
            # print(scores)
            # 1 / 0

            torch.save(scores, save_path)

            if counter % 20 == 0:
                time.sleep(60)


def omg_align_videos(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    splt = root_path.split('/')
    target_path = ('/').join(splt[:-3]) + '/Aligned256/' + splt[-2] + '/'
    print(f'Saving to {target_path}')
    root_dir = pathlib.Path(root_path)
    videos = [str(p) for p in list(root_dir.glob('*/'))
              if str(p).split('/')[-1] != '.DS_Store']
    assert len(videos) > 0

    for i_video, video in enumerate(tqdm(videos)):
        vid_name = video.split('/')[-1]
        utterances = [str(p) for p in list(pathlib.Path(video).glob('*/'))
                      if str(p).split('/')[-1] != '.DS_Store']

        for i_utt, utterance in enumerate(utterances):
            utt_name = utterance.split('/')[-1][:-4]
            save_dir = os.path.join(target_path, vid_name, utt_name)
            print("Video [{}/{}], Utterance [{}/{}], {}".format(
                i_video + 1, len(videos),
                i_utt + 1, len(utterances),
                save_dir))

            aligner = VideoAligner()
            aligner.align_video(utterance, save_dir)


def aff_wild2_align_videos(root_path, group):
    # Load landmarks model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    if root_path[-1] != '/':
        root_path += '/'

    splt = root_path.split('/')
    target_path = ('/').join(splt[:-3]) + '/Aligned256/' + splt[-2] + '/'
    print(f'Saving to {target_path}')
    root_dir = pathlib.Path(root_path)
    videos = [str(p) for p in list(root_dir.glob('*/'))
              if str(p).split('/')[-1].split('.')[-1] in ['mp4', 'avi']]
    assert len(videos) > 0

    groups = []
    n = len(videos) // 7
    for i in range(0, len(videos), n):
        groups.append(videos[i:i + n])

    videos = groups[group]
    print(f"Group {group}, num_videos {len(videos)}")

    for i_video, video in enumerate(tqdm(videos)):
        vid_name = video.split('/')[-1].split('.')[0]
        print("Video [{}/{}]".format(i_video + 1, len(videos)))
        path_to_vid = os.path.join(target_path, vid_name)

        os.makedirs(path_to_vid, exist_ok=True)

        # Restart frame counter
        i_frame = 0

        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            # Frame shape: (720, 1280, 3)
            ret, frame = cap.read()
            if not ret:
                break
            i_frame += 1
            save_str = os.path.join(
                path_to_vid, str(i_frame).zfill(3) + '.png')
            if os.path.exists(save_str):
                continue

            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pre-resize to save computation
            h_old, w_old, _ = frame.shape
            h_new = 256
            factor = h_new / h_old
            w_new = int(w_old * factor)
            frame_small = cv2.resize(frame, (w_new, h_new))

            # Grayscale image
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            # Detect faces
            for rect in detector(frame_small, 1):
                landmarks = [(int(item.x / factor), int(item.y / factor))
                             for item in predictor(gray_small, rect).parts()]
                frame = align_image(
                    frame, landmarks, output_size=256, transform_size=1024)
                frame.save(save_str)
                # print(save_str)
                # frame.show()
                # 1 / 0
                break


def omg_get_forced_alignment(root_path):
    all_audios = [str(a) for a in list(pathlib.Path(root_path).glob('*/*.wav'))]

    transcript_path = [str(a) for a in list(
        pathlib.Path(root_path).glob('*_transcripts.csv'))][0]
    transcripts = pd.read_csv(transcript_path)

    print(transcripts)
    print(len(all_audios))

    for audio in tqdm(all_audios):
        video, utterance = audio.split('/')[-2:]
        forced_alignment_path = audio[:-4] + '.json'
        if os.path.exists(forced_alignment_path):
            continue
        utterance = utterance[:-4] + '.mp4'
        transcript = transcripts.loc[(
            transcripts.video == video) & (transcripts.utterance == utterance)].transcript.to_list()[0]
        transcript = str(transcript)
        # print(transcript)
        # print(audio)
        # print(video, utterance)
        # print(forced_alignment_path)
        # 1 / 0
        print(audio, transcript)
        t_path = '../saves/tmp.txt'
        with open(t_path, 'w') as f:
            f.write(transcript)

        command = f'curl -F "audio=@{audio}" -F "transcript=@{t_path}" "http://localhost:8765/transcriptions?async=false"'

        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        # out = proc.stdout.read()
        out, err = proc.communicate()
        aligmnent = json.loads(out)

        with open(forced_alignment_path, 'w') as f:
            json.dump(aligmnent, f)


def omg_get_phoneme_timing(root_path):
    all_aligned = [str(a)
                   for a in list(pathlib.Path(root_path).glob('*/*.json'))]

    # aligned256_root = '/'.join(root_path.split('/')[:-3] + ['Aligned256'] + root_path.split('/')[-2:])

    info_path = [str(a) for a in list(
        pathlib.Path(root_path).glob('*_info.csv'))][0]
    info = pd.read_csv(info_path)

    print(len(all_aligned))

    results = {}
    for aligned in tqdm(all_aligned):

        # Get corresponding frame sequence
        target_path = '/'.join(aligned.split('/')
                               [:-4] + ['Aligned256'] + aligned.split('/')[-3:])[:-5] + '/'
        target_frames = sorted([str(p) for p in list(
            pathlib.Path(target_path).glob('*.png'))])
        print(len(target_frames), target_path)
        maxi = int(target_frames[-1].split('/')[-1].split('.')[0])

        # Get relevant paths
        video, utterance = aligned.split('/')[-2:]
        utterance = utterance[:-5] + '.mp4'

        # Get FPS
        fps = float(info.loc[(info['video'] == video) & (
            info['utterance'] == utterance)].fps.to_list()[0])

        with open(aligned) as json_file:
            data = json.load(json_file)
        print(aligned)
        print(data['transcript'])
        # print("")
        words = data['words']

        # Remove failed words
        new_words = []
        for i_word in range(len(words)):
            if 'start' in words[i_word].keys():
                new_words.append(words[i_word])
        words = new_words

        # Fill missing silence words
        words_new = []
        for i_word in range(len(words)):
            if i_word != 0 and words[i_word - 1]['end'] < words[i_word]['start']:
                word = {
                    'alignedWord': 'sil',
                    'end': words[i_word]['start'],
                    'phones': [
                        {
                            'duration': round(words[i_word]['start'] - words[i_word - 1]['end'], 2),
                            'phone': 'sil'
                        }],
                    'start': words[i_word - 1]['end'],
                    'word': 'sil'
                }
                words_new.append(word)
            words_new.append(words[i_word])
        words = words_new

        # Get absolute timings for each phoneme
        for word in words:
            phones = word['phones']
            for i in range(len(phones)):
                if i == 0:
                    phones[i]['abs_start'] = word['start']
                    phones[i]['abs_end'] = round(
                        phones[i]['abs_start'] + phones[i]['duration'], 2)
                else:
                    phones[i]['abs_start'] = phones[i - 1]['abs_end']
                    phones[i]['abs_end'] = round(
                        phones[i]['abs_start'] + phones[i]['duration'], 2)

        # Get timings in a dict
        timings = {}
        timings_f = {}
        for i_word in range(len(words)):
            # print(words[i_word]['word'])
            for phone in words[i_word]['phones']:
                timings[phone['abs_start']] = phone['phone']
                timings_f[phone['abs_start'] // (1. / fps)] = phone['phone']
                # print(phone['abs_start'], phone['abs_start'] // (1. / fps), timings[phone['abs_start']])
            if i_word == len(words) - 1:
                timings[phone['abs_end']] = 'sil'
                timings_f[phone['abs_end'] // (1. / fps)] = 'sil'
                # print(phone['abs_end'], phone['abs_end'] // (1. / fps), timings[phone['abs_end']])
            # print("")

        # Assign Phonemes to frames
        phone = 'sil'
        phones = []
        for i in range(maxi):
            target_frame = target_path + str(i + 1).zfill(3) + '.png'
            if i in timings_f.keys():
                phone = timings_f[i]
            phones.append(phone)
            # print(target_frame, i, phone)
            results[target_frame] = phone

        print("")

    with open(root_path + 'omg_mapping_phoneme.json', 'w') as f:
        json.dump(results, f)


def omg_extract_face_feats(root_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        HOME + '/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    if root_path[-1] != '/':
        root_path += '/'
    save_path = root_path + 'omg_mouth_features.npy'

    videos = [str(v) for v in list(pathlib.Path(root_path).glob('*/*/'))]

    all_paths = [sorted([str(p) for p in list(pathlib.Path(v).glob('*.png'))])
                 for v in videos]

    result = {}
    result = np.load(save_path, allow_pickle=True).item()

    counter = 0
    for v in tqdm(all_paths):
        for f in v:
            video, utterance, n_frame = f.split('/')[-3:]
            key = f"{video}/{utterance}/{n_frame}"
            if key in result.keys():
                continue
            # Load image
            frame = cv2.imread(f)

            # Grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect faces
            rects = detector(frame, 1)
            if len(list(rects)) == 0:
                print(f"WARNING. NO FACE DETECTED IN {f}")
                counter += 1
            for rect in rects:
                landmarks = [(int(item.x), int(item.y))
                             for item in predictor(gray, rect).parts()]
                landmarks = np.array(landmarks)

                params = get_mouth_params(landmarks, frame)

                # Visualize
                # for (x, y) in landmarks:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.imshow("Output", frame)
                # cv2.waitKey(0)
                # print(key)
                # 1 / 0

                result[key] = params
                break

    # Save params
    print(f"NO FACES DETECTED IN {counter} FRAMES")
    print("Done. Saving...")
    np.save(save_path, result)


def tagesschau_align_videos(root_path, group):
    if root_path[-1] != '/':
        root_path += '/'

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')
    videos = sorted(glob(root_path + '*.mp4'))
    assert len(videos) > 0

    groups = []
    n = len(videos) // 7
    for i in range(0, len(videos), n):
        groups.append(videos[i:i + n])

    videos = groups[group]
    print(f"Group {group}, num_videos {len(videos)}, {len(groups)} groups in total")

    for i_video, video in enumerate(tqdm(videos)):
        vid_name = video.split('/')[-1][:-4]
        save_dir = os.path.join(target_path, vid_name)
        print("Video [{}/{}], {}".format(
            i_video + 1, len(videos),
            save_dir))

        aligner = VideoAligner()
        aligner.align_video(video, save_dir)


def tagesschau_encode_frames(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    videos = sorted(glob(root_path + '*/'))
    videos = [sorted(glob(v + '*.png')) for v in videos]
    all_frames = [item for sublist in videos for item in sublist]
    assert len(all_frames) > 0
    print(len(all_frames))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoder
    from my_models.models import resnetEncoder
    e = resnetEncoder(net=18, pretrained=True).eval().to(device)

    # Get latent avg
    from my_models.style_gan_2 import PretrainedGenerator1024
    g = PretrainedGenerator1024().eval()
    latent_avg = g.latent_avg.view(1, -1).repeat(18, 1)

    # transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for frame in tqdm(all_frames):
        save_path = frame.split('.')[0] + '.latent.pt'
        # print(save_path)
        if os.path.exists(save_path):
            continue

        # Load image
        img = t(Image.open(frame)).unsqueeze(0).to(device)

        # Encoder image
        with torch.no_grad():
            latent_offset = e(img)[0].cpu()
            latent = latent_offset + latent_avg

        # Visualize
        from torchvision.utils import make_grid
        from utils.utils import downsample_256
        print(save_path, latent.shape)
        img_gen = g.to(device)([latent.to(device)], input_is_latent=True, noise=g.noises)[0].cpu()
        img_gen = downsample_256(img_gen)
        img_gen = make_grid(torch.cat((img_gen, img.cpu()), dim=0), normalize=True, range=(-1, 1))
        img_gen = transforms.ToPILImage('RGB')(img_gen)
        img_gen.show()
        1 / 0

        # Save
        # torch.save(latent, save_path)


def tagesschau_encode_frames_center(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    videos = sorted(glob(root_path + '*/'))
    videos = [sorted(glob(v + '*.png')) for v in videos]
    all_frames = [item for sublist in videos for item in sublist]
    assert len(all_frames) > 0
    print(len(all_frames))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoder
    from my_models.models import resnetEncoder
    e = resnetEncoder(net=18, pretrained=True).eval().to(device)

    # Get latent avg
    from my_models.style_gan_2 import PretrainedGenerator1024
    g = PretrainedGenerator1024().eval().to(device)

    # transforms
    t_load = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    t_flip = transforms.RandomHorizontalFlip(1.0)

    for frame in tqdm(all_frames):
        save_path = frame.split('.')[0] + '.center_latent.pt'
        # print(save_path)
        if os.path.exists(save_path):
            continue

        # Load image
        pil_img = Image.open(frame)
        img = t_load(pil_img).to(device)
        img_flip = t_load(t_flip(pil_img)).to(device)

        # Encoder image
        with torch.no_grad():
            enc = e(img.unsqueeze(0))[0] + g.latent_avg
            enc_flip = e(img_flip.unsqueeze(0))[0] + g.latent_avg
            latent = 0.5 * (enc + enc_flip)
            latent = latent.cpu()

        # Visualize
        from torchvision.utils import make_grid
        print(save_path, latent.shape)
        img_gen = g.to(device)([latent.to(device)],
                               input_is_latent=True, noise=g.noises)[0].cpu()
        img_gen = make_grid(torch.cat((img_gen, img.cpu()),
                                      dim=0), normalize=True, range=(-1, 1))
        img_gen = transforms.ToPILImage('RGB')(img_gen)
        img_gen.show()
        1 / 0

        # Save
        # torch.save(latent, save_path)


def tagesschau_get_mean_latents(root):
    # Load paths
    videos = sorted(glob(root + '*/'))

    for video in tqdm(videos):
        latent_paths = sorted(glob(video + '*.latent.pt'))

        mean_latent = []
        for latent_path in latent_paths:
            latent = torch.load(latent_path).unsqueeze(0)
            mean_latent.append(latent)
        mean_latent = torch.cat(mean_latent, dim=0).mean(dim=0)

        # Save
        torch.save(mean_latent, video + 'mean.latent.pt')


def tagesschau_get_landmarks(root, group):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat')

    videos = sorted(glob(root + '*/'))
    frames = [sorted(glob(v + '*.png')) for v in videos]
    frames = [item for sublist in frames for item in sublist]

    groups = []
    n = len(frames) // 3
    for i in range(0, len(frames), n):
        groups.append(frames[i:i + n])

    frames = groups[group]
    print(
        f"Group {group}, num_frames {len(frames)}, {len(groups)} groups in total")

    for frame in tqdm(frames):
        save_path = frame.split('.')[0] + '.landmarks.pt'
        if os.path.exists(save_path):
            continue
        img = io.imread(frame)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces
        rects = detector(img, 1)
        for (i, rect) in enumerate(rects):
            # Detect landmarks in faces
            landmarks = predictor(gray, rect)
            landmarks = torch.tensor(shape_to_np(landmarks))

            # Visualize
            # print(save_path)
            # for (x, y) in landmarks:
            #     cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
            # cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # 1 / 0

            # Save
            torch.save(landmarks, save_path)

            break


def tagesschau_gather_info(root):
    save_path = root + 'latent_data.pt'
    frames = sorted(glob(root + '*/*.png'))

    names = []
    latents = []
    landmarks = []
    deepspeechs = []
    for frame in tqdm(frames):
        base = frame.split('.')[0]
        name = '/'.join(base.split('/')[-2:])
        latent = torch.load(base + '.latent.pt')
        landmark = torch.load(base + '.landmarks.pt')
        deepspeech = torch.tensor(
            np.load(base + '.deepspeech.npy'), dtype=torch.float32)

        names.append(name)
        latents.append(latent)
        landmarks.append(landmark)
        deepspeechs.append(deepspeech)

    names = np.array(names)
    latents = torch.stack(latents)
    landmarks = torch.stack(landmarks)
    deepspeechs = torch.stack(deepspeechs)

    data = {
        'names': names,
        'latents': latents,
        'landmarks': landmarks,
        'deepspeech': deepspeechs
    }

    torch.save(data, save_path)


def ravdess_get_3D_landmarks(root):
    if root[-1] != '/':
        root += '/'

    # Init face alignment
    import face_alignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=False, device='cuda')

    # Get all sentences in a list
    sentences = sorted(glob(root + '*/*/'))

    for sentence in tqdm(sentences):
        lm = fa.get_landmarks_from_directory(sentence)

        for key, value in lm.items():
            save_path = key.split('.')[0] + '.landmarks3d.pt'
            value = torch.tensor(value, dtype=torch.float32)
            if len(value.shape) == 3:
                value = value[0]

            # Visualize
            # print(save_path)
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure(figsize=(5, 5))
            # x, y, z = value.unbind(1)
            # ax = Axes3D(fig)
            # ax.scatter3D(x, -z, y)
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # ax.view_init(180, 90)
            # plt.show()
            # 1 / 0

            # Save
            torch.save(value, save_path)


def tagesschau_get_mouth_mask():
    def get_face_mask(mouth, image_size=256):
        mask = np.zeros((image_size, image_size, 3), dtype='uint8')
        mask = cv2.drawContours(mask, [mouth], -1,
                                (255, 255, 255), thickness=cv2.FILLED)
        mask = cv2.bitwise_not(mask)
        img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
        return mask

    def display(mouth):
        img = np.zeros((256, 256, 3))
        for (x, y) in mouth:
            cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
        cv2.imshow("", img)
        cv2.waitKey(0)

    data = torch.load(
        '/home/meissen/Datasets/Tagesschau/Aligned256/latent_data.pt')
    lms = data['landmarks']

    left = lms[:, 48]
    upper_left = lms[:, 49:51]
    top = lms[:, 51]
    upper_right = lms[:, 52:54]
    right = lms[:, 54]
    lower_right = lms[:, 55:57]
    down = lms[:, 57]
    lower_left = lms[:, 58:60]

    # Version 1 max mouth
    # m_left = torch.tensor([[left[:, 0].min(), left[:, 1].float().mean()]], dtype=torch.long)
    # m_upper_left = torch.stack((upper_left[:, :, 0].min(0)[0], upper_left[:, :, 1].min(0)[0])).t()
    # m_top = torch.tensor([[top[:, 0].float().mean(), top[:, 1].min()]], dtype=torch.long)
    # m_upper_right = torch.stack((upper_right[:, :, 0].max(0)[0], upper_right[:, :, 1].min(0)[0])).t()
    # m_right = torch.tensor([[right[:, 0].max(), right[:, 1].float().mean()]], dtype=torch.long)
    # m_lower_right = torch.stack((lower_right[:, :, 0].max(0)[0], lower_right[:, :, 1].max(0)[0])).t()
    # m_down = torch.tensor([[down[:, 0].float().mean(), down[:, 1].max()]], dtype=torch.long)
    # m_lower_left = torch.stack((lower_left[:, :, 0].min(0)[0], lower_left[:, :, 1].max(0)[0])).t()

    # mouth = torch.cat((m_left, m_upper_left, m_top, m_upper_right, m_right, m_lower_right, m_down, m_lower_left))
    # mask = get_face_mask(mouth.numpy(), 256)
    # mask = Image.fromarray(mask)
    # mask.show()
    # mask = transforms.ToTensor()(mask)
    # torch.save(mask, '/home/meissen/Datasets/Tagesschau/Aligned256/mask_max_mouth.pt')

    # Version 2 std mouth
    STD = 5.
    # Left
    left = left.float()
    s_leftx = left[:, 0].mean() - STD * left[:, 0].std()
    s_lefty = left[:, 1].mean()
    s_left = torch.tensor([[s_leftx, s_lefty]])
    # Upper left
    ul = upper_left.float()
    s_ulx = ul[:, :, 0].mean(0) - STD * ul[:, :, 0].std(0)
    s_uly = ul[:, :, 1].mean(0) - STD * ul[:, :, 1].std(0)
    s_ul = torch.stack((s_ulx, s_uly)).t()
    # Top
    top = top.float()
    s_topx = top[:, 0].mean()
    s_topy = top[:, 1].mean() - STD * top[:, 1].std()
    s_top = torch.tensor([[s_topx, s_topy]])
    # Upper right
    ur = upper_right.float()
    s_urx = ur[:, :, 0].mean(0) + STD * ur[:, :, 0].std(0)
    s_ury = ur[:, :, 1].mean(0) - STD * ur[:, :, 1].std(0)
    s_ur = torch.stack((s_urx, s_ury)).t()
    # Right
    right = right.float()
    s_rightx = right[:, 0].mean() + STD * right[:, 0].std()
    s_righty = right[:, 1].mean()
    s_right = torch.tensor([[s_rightx, s_righty]])
    # Lower right
    lr = lower_right.float()
    s_lrx = lr[:, :, 0].mean(0) + STD * lr[:, :, 0].std(0)
    s_lry = lr[:, :, 1].mean(0) + STD * lr[:, :, 1].std(0)
    s_lr = torch.stack((s_lrx, s_lry)).t()
    # Down
    down = down.float()
    s_downx = down[:, 0].mean()
    s_downy = down[:, 1].mean() + STD * down[:, 1].std()
    s_down = torch.tensor([[s_downx, s_downy]])
    # Lower left
    ll = lower_left.float()
    s_llx = ll[:, :, 0].mean(0) - STD * ll[:, :, 0].std(0)
    s_lly = ll[:, :, 1].mean(0) + STD * ll[:, :, 1].std(0)
    s_ll = torch.stack((s_llx, s_lly)).t()

    mouth = torch.cat((s_left, s_ul, s_top, s_ur, s_right, s_lr, s_down, s_ll))

    mask = get_face_mask(np.around(mouth.numpy()).astype('int'), 256)
    mask = Image.fromarray(mask)
    mask.show()
    mask = transforms.ToTensor()(mask)

    # display(mouth)

    torch.save(
        mask, '/home/meissen/Datasets/Tagesschau/Aligned256/mask_5std_mouth.pt')


def get_landmarks(root_path, group):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat')

    if root_path[-1] != '/':
        root_path += '/'

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')
    videos = sorted(glob(root_path + '*/'))
    assert len(videos) > 0

    groups = []
    n = len(videos) // 6
    for i in range(0, len(videos), n):
        groups.append(videos[i:i + n])

    videos = groups[group]
    print(
        f"Group {group}, num_videos {len(videos)}, {len(groups)} groups in total")

    for video in tqdm(videos):
        print(video)
        frames = sorted(glob(video + '*.png'))

        for frame in frames:
            save_path = frame.split('.')[0] + '.landmarks.pt'
            img = io.imread(frame)

            # Grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Detect faces
            rects = detector(img, 1)
            if len(rects) == 0:
                print(f"Did not detect a face in {frame}")
            for rect in rects:
                landmarks = torch.tensor([(item.x, item.y) for item in predictor(
                    gray, rect).parts()], dtype=torch.float32)

                # Visualize
                # print(save_path)
                # print(landmarks.shape)
                # for (x, y) in landmarks:
                #     cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
                # cv2.imshow("", img)
                # cv2.waitKey(0)
                # 1 / 0

                # Save
                torch.save(landmarks, save_path)


def get_3ddfa_params(root_path):
    device = 'cuda'

    model = dreiDDFA().to(device)

    videos = sorted(glob(root_path + '*/'))
    frames = [sorted(glob(v + '*.png')) for v in videos]
    frames = [item for sublist in frames for item in sublist]

    t = transforms.ToTensor()

    for frame in tqdm(frames):
        save_path = frame.split('.')[0] + '.3ddfa.pt'
        if os.path.exists(save_path):
            continue

        img = t(Image.open(frame)).unsqueeze(0).to(device)
        res = model.predict_param(img)
        if res is None:
            print(f"No face detected in {frame}")
            continue

        params = {
            'param': res['param'][0].cpu(),
            'roi_box': res['roi_boxes']
        }

        # Visualize
        # print(params['param'].shape)
        # print(params['roi_box'])
        # print(save_path)
        # from dreiDDFA.ddfa_utils import draw_landmarks
        # landmarks = model.reconstruct_vertices(res['param'], res['roi_boxes'], dense=False)
        # draw_landmarks(img[0].cpu(), landmarks[0].cpu(), show_flg=True)
        # 1 / 0

        # Save
        torch.save(params, save_path)


if __name__ == "__main__":

    path = sys.argv[1]
    tagesschau_encode_frames_center(path)
