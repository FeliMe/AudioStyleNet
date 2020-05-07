import argparse
import cv2
import numpy as np
import os
import sys
import torch

from eafa import Emotion_Aware_Facial_Animation
from glob import glob
from PIL import Image
from tqdm import tqdm
from utils.alignment_handler import AlignmentHandler
from utils.psnr import PSNR
from utils.ssim import SSIM


def load_video(videofile):
    frames = []
    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.array(frames)


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def compute_metric(prediction, target, metric_fn):
    aligner = AlignmentHandler()
    i_frame = 0
    metric_arr = []
    for frame_pred, frame_target in zip(prediction, target):
        i_frame += 1
        lm_pred = aligner.get_landmarks(frame_pred)
        lm_target = aligner.get_landmarks(frame_target)
        if lm_pred is None:
            print(
                f"Did not find a face in prediction frame {i_frame}, skipping")
            continue
        if lm_target is None:
            print(f"Did not find a face in target frame {i_frame}, skipping")
            continue

        aligned_pred = aligner.align_face_static(
            frame_pred, lm_pred, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(163, 163))[0]
        aligned_target = aligner.align_face_static(
            frame_target, lm_target, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(163, 163))[0]

        # Visualize
        # Image.fromarray(aligned_pred).show()
        # Image.fromarray(aligned_target).show()
        # 1 / 0

        aligned_pred = np2torch_img(aligned_pred)
        aligned_target = np2torch_img(aligned_target)

        metric = metric_fn(aligned_pred, aligned_target)
        metric_arr.append(metric)

    metric_arr = np.array(metric_arr)
    return metric_arr


if __name__ == '__main__':

    # Init model
    model = Emotion_Aware_Facial_Animation(
        model_path=sys.argv[1],
        model_type='net3',
        audio_type='deepspeech',
        T=8,
        n_latent_vec=4,
        normalize_audio=False
    )

    root_path = '/mnt/sdb1/meissen/Datasets/GRID/'
    latent_root = root_path + 'Aligned256/'
    target_root = root_path + 'Video/'

    metric_name = 'ssim'
    if metric_name.lower() == 'psnr':
        metric_fn = PSNR()
    elif metric_name.lower() == 'ssim':
        metric_fn = SSIM()
    else:
        raise NotImplementedError

    videos = []
    with open(root_path + 'grid_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    metric_sum = 0.
    pbar = tqdm(total=len(videos))
    for video in videos:
        latentfile = sorted(glob(f"{latent_root}{video}/mean.latent.pt"))
        sentence = f"{latent_root}{video}/"
        targetfile = f"{target_root}{video}.mpg"
        # print(f"Image {imagefile} - Audio {audiofile} - Target {targetfile}")

        # Create video
        vid = model(test_latent=latentfile, test_sentence_path=sentence)
        vid = ((np.rollaxis(vid.numpy(), 1, 4) + 1) * 127.5).astype(np.uint8)

        # Compute metric
        target = load_video(targetfile)
        metric = compute_metric(vid, target, metric_fn)
        metric_sum += metric.mean()
        pbar.update()
        pbar.set_description(f"{metric_name}: {metric.mean():.4f}")

    print(f"mean {metric_name}: {metric_sum / len(videos):.4f}")
    print(f"prediction was {root_path}")
