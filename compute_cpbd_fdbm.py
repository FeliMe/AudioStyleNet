import cv2
import numpy as np
import os
import sys
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


def compute_metric(prediction, metric_fn):
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
        # Image.fromarray(aligned_pred).show()
        # 1 / 0

        aligned_pred = cv2.cvtColor(aligned_pred, cv2.COLOR_RGB2GRAY)

        metric = metric_fn(aligned_pred)
        metric_arr.append(metric)

    metric_arr = np.array(metric_arr)
    return metric_arr


if __name__ == '__main__':

    # Init model
    model = Emotion_Aware_Facial_Animation(
        model_path=sys.argv[1],
        device='cuda:2',
        model_type='net3',
        audio_type='deepspeech',
        T=8,
        n_latent_vec=4,
        normalize_audio=False
    )

    root_path = RAIDROOT + 'Datasets/GRID/'
    latent_root = root_path + 'Aligned256/'
    target_root = root_path + 'Video/'

    metric_name = 'fdbm'
    if metric_name.lower() == 'cpbd':
        metric_fn = compute
    elif metric_name.lower() == 'fdbm':
        metric_fn = FDBM()
    else:
        raise NotImplementedError

    videos = []
    with open(root_path + 'grid_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    metric_mean = 0.
    pbar = tqdm(total=len(videos))
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        targetfile = f"{target_root}{video}.mpg"
        # print(f"Image {imagefile} - Audio {audiofile})

        # Create video
        vid = model(test_latent=latentfile, test_sentence_path=sentence)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        metric = compute_metric(vid, metric_fn)
        metric_mean += metric.mean()
        pbar.update()
        pbar.set_description(f"{metric_name}: {metric.mean():.4f}")

    print(f"mean {metric_name}: {metric_mean / len(videos):.4f}")
    print(f"prediction was {root_path}")
