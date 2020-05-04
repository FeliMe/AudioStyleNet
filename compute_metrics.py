import argparse
import cv2
import numpy as np
import os
import sys
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm
from utils.alignment_handler import AlignmentHandler
from utils.psnr import PSNR
from utils.ssim import SSIM


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def video_metric(prediction_video, target_video, metric_fn):
    """
    Computes peak signal to noise ratio between two videos. Faces are being aligned
    """
    aligner = AlignmentHandler()
    i_frame = 0
    cap_pred = cv2.VideoCapture(prediction_video)
    cap_target = cv2.VideoCapture(target_video)
    metric_arr = []
    while cap_pred.isOpened() and cap_target.isOpened():
        # Frame shape: (weight, width, 3)
        ret_pred, frame_pred = cap_pred.read()
        ret_target, frame_target = cap_target.read()
        if not ret_pred or not ret_target:
            break

        i_frame += 1

        frame_pred = cv2.cvtColor(frame_pred, cv2.COLOR_BGR2RGB)
        frame_target = cv2.cvtColor(frame_target, cv2.COLOR_BGR2RGB)

        lm_pred = aligner.get_landmarks(frame_pred)
        lm_target = aligner.get_landmarks(frame_target)
        if lm_pred is None:
            print(f"Did not find a face in prediction frame {i_frame}, skipping")
            continue
        if lm_target is None:
            print(f"Did not find a face in target frame {i_frame}, skipping")
            continue

        aligned_pred = aligner.align_face_static(
            frame_pred, lm_pred, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(163, 163))[0]
        aligned_target = aligner.align_face_static(
            frame_target, lm_target, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(163, 163))[0]

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

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str,
                        default='/mnt/sdb1/meissen/Datasets/GRID/results_lpips_latent_mse_100k_mult3/')
    parser.add_argument('--target_path', type=str,
                        default='/mnt/sdb1/meissen/Datasets/GRID/Video/')
    parser.add_argument('--metric_name', type=str, default='psnr')  # 'psnr' or 'ssim'
    args = parser.parse_args()

    # Prediction videos
    if os.path.isdir(args.prediction_path):
        prediction_files = glob(args.prediction_path + '*.mp4')
        prediction_files += glob(args.prediction_path + '*.mov')
        prediction_files += glob(args.prediction_path + '*.avi')
    else:
        prediction_files = [args.prediction_path]
    prediction_files = sorted(prediction_files)

    # Target videos
    if os.path.isdir(args.target_path):
        target_files = glob(args.target_path + '*.mpg')
        target_files += glob(args.target_path + '*.mp4')
    else:
        target_files = [args.target_path]
    target_files = sorted(target_files)

    assert (len(prediction_files) != 0) and (len(target_files) != 0)
    print(f"{len(prediction_files)} videos")

    if args.metric_name.lower() == 'psnr':
        metric_fn = PSNR()
    elif args.metric_name.lower() == 'ssim':
        metric_fn = SSIM()
    else:
        raise NotImplementedError

    metric_sum = 0.
    pbar = tqdm(total=len(prediction_files))
    for prediction_video, target_video in zip(prediction_files, target_files):
        metric = video_metric(prediction_video, target_video, metric_fn)
        metric_sum += metric.mean()
        pbar.update()
        pbar.set_description(f"{args.metric_name}: {metric.mean():.4f}")
        # print(prediction_video, target_video)

    print(f"mean {args.metric_name}: {metric_sum / len(prediction_files):.4f}")
    print(f"prediction was {args.prediction_path}")
