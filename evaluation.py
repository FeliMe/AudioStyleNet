"""
Evaluate a model on different metrics and datasets
"""

import argparse
import cv2
import numpy as np
import os
import torch

from audiostylenet import AudioStyleNet
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.alignment_handler import AlignmentHandler
from utils import metrics, lipnet
from utils.utils import downsample_256


RAIDROOT = os.environ.get('RAIDROOT')


def load_video(videofile):
    frames = []
    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert len(frames) > 0, f"Failed to load video {videofile}"
    return np.array(frames)


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def image_from_latent(latentfile, eafa_model):
    latent = torch.load(latentfile).unsqueeze(0).cuda()
    with torch.no_grad():
        img = eafa_model.g([latent], input_is_latent=True,
                           noise=eafa_model.g.noises)[0].cpu()
    img = downsample_256(img)
    img = make_grid(img, normalize=True, range=(-1, 1))
    return img


def compute_psnr_ssim(model, videos, metric_name, verbose=False):

    def compute_metric(prediction, target, metric_fn, verbose=False):
        aligner = AlignmentHandler(detector='frontal')
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
                print(
                    f"Did not find a face in target frame {i_frame}, skipping")
                continue

            aligned_pred = aligner.align_face_static(
                frame_pred, lm_pred, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(128, 128))[0]
            aligned_target = aligner.align_face_static(
                frame_target, lm_target, desiredLeftEye=(0.28, 0.23), desiredFaceShape=(128, 128))[0]

            # Visualize
            if verbose:
                Image.fromarray(aligned_pred).show()
                Image.fromarray(aligned_target).show()
                1 / 0

            aligned_pred = np2torch_img(aligned_pred)
            aligned_target = np2torch_img(aligned_target)

            metric = metric_fn(aligned_pred, aligned_target)
            metric_arr.append(metric)

        metric_arr = np.array(metric_arr)
        return metric_arr

    if metric_name.lower() == 'psnr':
        metric_fn = metrics.PSNR()
    elif metric_name.lower() == 'ssim':
        metric_fn = metrics.SSIM()
    else:
        raise NotImplementedError

    metric_mean = []
    pbar = tqdm(total=len(videos))
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        targetfile = f"{target_root}{video}{video_ext}"
        # print(f"Image {imagefile} - Audio {audiofile} - Target {targetfile}")

        # Create video
        max_sec = 30 if dataset == 'AudioVisualDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        target = load_video(targetfile)
        metric = compute_metric(vid, target, metric_fn, verbose=args.verbose)
        metric_mean.append(metric.mean())
        pbar.update()
        pbar.set_description(
            f"{metric_name}: {metric.mean():.4f} - current mean: {np.array(metric_mean).mean():.4f}")

    print(f"mean {metric_name}: {np.array(metric_mean).mean():.4f}")
    print(f"prediction was {root_path}")


def compute_facenet_dist(model, videos, verbose=False):

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

    metric_fn = metrics.FaceNetDist(device=device, image_size=109)
    metric_name = 'facenet_dist'
    metric_mean = []
    pbar = tqdm(total=len(videos))
    for video in videos:
        pbar.update()
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"

        static_image = image_from_latent(latentfile, model)
        static_image = (static_image.permute(
            1, 2, 0).numpy() * 255.).astype(np.uint8)

        # Create video
        max_sec = 30 if dataset == 'AudioVisualDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        metric = compute_metric(
            vid, static_image, metric_fn, verbose=args.verbose)
        if metric is None:
            continue
        metric_mean.append(metric.mean())
        pbar.set_description(
            f"{metric_name}: {metric.mean():.4f} - current mean: {np.array(metric_mean).mean():.4f}")

    print(f"mean {metric_name}: {np.array(metric_mean).mean():.4f}")
    print(f"prediction was {root_path}")


def compute_lipnet_wer(model, videos, device, verbose=False):
    from jiwer import wer, mer, wil
    lipnet_model = lipnet.get_model(device)
    wer_sum = 0.
    mer_sum = 0.
    wil_sum = 0.
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        transcriptfile = f"{transcript_root}{video}.transcript.txt"

        # Load transcript
        transcript = lipnet.read_transcript(transcriptfile)

        # Create video
        max_sec = 30 if dataset == 'AudioVisualDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        vid = lipnet.prepare_video(vid, device, verbose=args.verbose)

        prediction = lipnet.lipnet_predict(vid, lipnet_model)
        if prediction is None:
            continue
        transcript = lipnet.read_transcript(transcriptfile)
        wer_error = wer(transcript, prediction)
        mer_error = mer(transcript, prediction)
        wil_error = wil(transcript, prediction)
        wer_sum += wer_error
        mer_sum += mer_error
        wil_sum += wil_error
        print(f"WER {wer_error:.4f} - MER {mer_error:.4f} - WIL {wil_error:.4f} prediction: {prediction} | transcript: {transcript}")

    print(f"Mean WER {wer_sum / len(videos):.4f}")
    print(f"Mean MER {mer_sum / len(videos):.4f}")
    print(f"Mean WIL {wil_sum / len(videos):.4f}")


def run_dataset(model, videos, verbose=False):
    pbar = tqdm(total=len(videos))
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        audiofile = f"{audio_root}{video}.wav"

        pbar.set_description(video)

        # Create video
        max_sec = 30 if dataset == 'AudioVisualDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)

        # Visualize
        if args.verbose:
            from torchvision import transforms
            transforms.ToPILImage()(vid[0]).show()
            1 / 0

        # Save video
        model.save_video(vid, audiofile, f"{target_path}{video}.avi")

        pbar.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--audio_multiplier', type=float, default=2.0)
    parser.add_argument('--audio_truncation', type=float, default=0.8)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Init model
    model = AudioStyleNet(
        model_path=args.model_path,
        device=device,
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
    audio_root = root_path + 'Audio/'
    transcript_root = root_path + 'Video/'

    if dataset == 'GRID':
        video_ext = '.mpg'
    elif dataset == 'CREMA-D':
        video_ext = '.flv'
    elif dataset == 'AudioVisualDataset':
        video_ext = '.mp4'
    else:
        raise NotImplementedError

    videos = []
    with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    if args.metric in ['psnr', 'ssim']:
        compute_psnr_ssim(model, videos, args.metric, args.verbose)
    elif args.metric == 'facenet_dist':
        compute_facenet_dist(model, videos, args.verbose)
    elif args.metric == 'lipnet_wer':
        assert args.dataset.lower() == 'grid', 'lipnet_wer is only available for the GRID dataset'
        compute_lipnet_wer(model, videos, device, args.verbose)
    elif args.metric is None:
        target_path = f'{root_path}results_own_model_{args.model_path.split("/")[-3]}/'
        os.makedirs(target_path, exist_ok=True)
    else:
        raise NotImplementedError("Unknown metric")
