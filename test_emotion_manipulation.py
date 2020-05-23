import argparse
import numpy as np
import os
import torch

from eafa import Emotion_Aware_Facial_Animation
from my_models.models import EmotionClassifier, FERClassifier


RAIDROOT = os.environ['RAIDROOT']

MAPPING = {
    'none': -1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgusted': 6,
    'surprised': 7
}


class VideoClassifier:
    def __init__(self, device, model):
        self.device = device
        if model == 'rav':
            self.classifier = EmotionClassifier(
                softmaxed=True).eval().to(self.device)
        elif model == 'fer':
            self.classifier = FERClassifier(
                softmaxed=True).eval().to(self.device)
        else:
            raise NotImplementedError

    def __call__(self, video):
        video = video.to(self.device)
        with torch.no_grad():
            scores = self.classifier(video)
        scores = scores.mean(dim=0)
        return scores


class VideoClassifier2:
    def __init__(self, device, model):
        self.device = device
        if model == 'rav':
            self.classifier = EmotionClassifier(
                softmaxed=True).eval().to(self.device)
        elif model == 'fer':
            self.classifier = FERClassifier(
                softmaxed=True).eval().to(self.device)
        else:
            raise NotImplementedError

    def __call__(self, video):
        video = video.to(self.device)
        with torch.no_grad():
            scores = self.classifier(video)
        scores = torch.argmax(scores, dim=1)
        unique, counts = torch.unique(scores, return_counts=True)
        res = torch.zeros(8)
        for index, count in zip(unique, counts):
            res[index] = count
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--test_nr', type=int, choices=[1, 2])
    parser.add_argument('--classification_model', type=str, choices=['rav', 'fer'])
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--model_type', type=str, default='net3')
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--audio_multiplier', type=float, default=2.0)
    parser.add_argument('--audio_truncation', type=float, default=0.8)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    np.set_printoptions(precision=4)

    # Init model
    model = Emotion_Aware_Facial_Animation(
        model_path=args.model_path,
        device=device,
        model_type=args.model_type,
        audio_type=args.audio_type,
        T=8,
        n_latent_vec=4,
        normalize_audio=False
    )

    if args.test_nr == 1:
        classifier = VideoClassifier(device, args.classification_model)
    elif args.test_nr == 2:
        classifier = VideoClassifier2(device, args.classification_model)
    else:
        raise NotImplementedError

    dataset = args.dataset

    if os.path.exists(f'/home/meissen/Datasets/{dataset}/'):
        root_path = f'/home/meissen/Datasets/{dataset}/'
    else:
        root_path = RAIDROOT + f'Datasets/{dataset}/'
    latent_root = root_path + 'Aligned256/'
    target_root = root_path + 'Video/'

    videos = []
    with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    for emotion in MAPPING.keys():
        if emotion == 'none':
            directionfile = None
        else:
            directionfile = f'saves/control_latent/directions/{emotion}_rav_lin.npy'

        dataset_scores = []
        for video in videos:
            latentfile = f"{latent_root}{video}/mean.latent.pt"
            sentence = f"{latent_root}{video}/"
            targetfile = f"{target_root}{video}.mpg"
            # print(f"Image {imagefile} - Audio {audiofile})

            # Create video
            max_sec = 30 if dataset == 'AudioDataset' else -1
            max_sec = 1 if args.verbose else max_sec
            vid = model(test_latent=latentfile, test_sentence_path=sentence,
                        direction=directionfile,
                        audio_multiplier=args.audio_multiplier,
                        audio_truncation=args.audio_truncation,
                        max_sec=max_sec)

            # Visualize
            if args.verbose:
                from torchvision import transforms
                transforms.ToPILImage()(vid[0]).save('/home/meissen/workspace/verbose.png')
                print("Showing result")
                1 / 0

            # Compute classification score
            video_scores = classifier(vid)
            dataset_scores.append(video_scores.cpu())

        dataset_scores = torch.stack(dataset_scores, dim=0)
        print(f"{emotion} scores {dataset_scores.mean(dim=0).numpy()}")
