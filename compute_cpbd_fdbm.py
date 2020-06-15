import argparse
import cv2
import numpy as np
import os
import torch

from utils.alignment_handler import AlignmentHandler
from cpbd import compute
from eafa import Emotion_Aware_Facial_Animation
from glob import glob
from utils.metrics import FDBM
from PIL import Image
from tqdm import tqdm


RAIDROOT = os.environ['RAIDROOT']


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def compute_metric(prediction, metric_fn, verbose=False):
    aligner = AlignmentHandler()
    i_frame = 0
    metric_arr = []
    for frame_pred in prediction:
        i_frame += 1
        lm_pred = aligner.get_landmarks(frame_pred)
        if lm_pred is None:
            print(
                f"Did not find a face in prediction frame {i_frame}, skipping")
            continue

        aligned_pred = aligner.align_face_static(
            frame_pred, lm_pred, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(128, 128))[0]

        # Visualize
        if verbose:
            Image.fromarray(aligned_pred).show()
            1 / 0

        aligned_pred = cv2.cvtColor(aligned_pred, cv2.COLOR_RGB2GRAY)

        metric = metric_fn(aligned_pred)
        metric_arr.append(metric)

    metric_arr = np.array(metric_arr)
    return metric_arr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--model_type', type=str, default='net3')
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--audio_multiplier', type=float, default=2.0)
    parser.add_argument('--audio_truncation', type=float, default=0.8)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Init model
    model = Emotion_Aware_Facial_Animation(
        model_path=args.model_path,
        device=device,
        model_type=args.model_type,
        audio_type=args.audio_type,
        T=8
    )

    dataset = args.dataset

    if os.path.exists(f'/home/meissen/Datasets/{dataset}/'):
        root_path = f'/home/meissen/Datasets/{dataset}/'
    else:
        root_path = RAIDROOT + f'Datasets/{dataset}/'
    latent_root = root_path + 'Aligned256/'
    target_root = root_path + 'Video/'

    metric_name = args.metric
    if metric_name.lower() == 'cpbd':
        metric_fn = compute
    elif metric_name.lower() == 'fdbm':
        metric_fn = FDBM()
    else:
        raise NotImplementedError

    videos = []
    with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    metric_mean = []
    pbar = tqdm(total=len(videos))
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        targetfile = f"{target_root}{video}.mpg"
        # print(f"Image {imagefile} - Audio {audiofile})

        # Create video
        max_sec = 30 if dataset == 'AudioDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        metric = compute_metric(vid, metric_fn, verbose=args.verbose)
        metric_mean.append(metric.mean())
        pbar.update()
        pbar.set_description(
            f"{metric_name}: {metric.mean():.4f} - current mean: {np.array(metric_mean).mean():.4f}")

    print(f"mean {metric_name}: {np.array(metric_mean).mean():.4f}")
    print(f"prediction was {root_path}")
