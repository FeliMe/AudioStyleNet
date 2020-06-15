import argparse
import numpy as np
import os
import sys
import torch

from eafa import Emotion_Aware_Facial_Animation
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.alignment_handler import AlignmentHandler
from utils.metrics import FaceNetDist
from utils.utils import downsample_256


RAIDROOT = os.environ['RAIDROOT']


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def image_from_latent(latentfile, eafa_model):
    latent = torch.load(latentfile).unsqueeze(0).cuda()
    with torch.no_grad():
        img = eafa_model.g([latent], input_is_latent=True, noise=eafa_model.g.noises)[0].cpu()
    img = downsample_256(img)
    img = make_grid(img, normalize=True, range=(-1, 1))
    return img


def compute_metric(prediction, static_image, metric_fn, verbose=False):
    aligner = AlignmentHandler()

    # Align static image
    lm_static = aligner.get_landmarks(static_image)
    if lm_static is None:
        print(f"Did not find a face in static image, skipping video")
        return None
    aligned_static = aligner.align_face_static(
        static_image, lm_static, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(128, 128))[0]
    aligned_static = Image.fromarray(aligned_static)

    # Loop over predicted video
    metric_arr = []
    for i_frame, frame_pred in enumerate(prediction):
        lm_pred = aligner.get_landmarks(frame_pred)
        if lm_pred is None:
            print(
                f"Did not find a face in prediction frame {i_frame}, skipping")
            continue

        aligned_pred = aligner.align_face_static(
            frame_pred, lm_pred, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(128, 128))[0]
        aligned_pred = Image.fromarray(aligned_pred)

        metric = metric_fn(aligned_pred, aligned_static, verbose)
        if metric is None:
            continue
        metric_arr.append(metric)

    if len(metric_arr) == 0:
        print(f"Video failed")
        return None

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

    metric_name = 'face_net_dist'
    metric_fn = FaceNetDist(device=device, image_size=109)

    videos = []
    with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    metric_mean = []
    pbar = tqdm(total=len(videos))
    for video in videos:
        pbar.update()
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        # print(f"Image {imagefile} - Audio {audiofile})

        static_image = image_from_latent(latentfile, model)
        static_image = (static_image.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)

        # Create video
        max_sec = 30 if dataset == 'AudioDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        metric = compute_metric(vid, static_image, metric_fn, verbose=args.verbose)
        if metric is None:
            continue
        metric_mean.append(metric.mean())
        pbar.set_description(
            f"{metric_name}: {metric.mean():.4f} - current mean: {np.array(metric_mean).mean():.4f}")

    print(f"mean {metric_name}: {np.array(metric_mean).mean():.4f}")
    print(f"prediction was {root_path}")
