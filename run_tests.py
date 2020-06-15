import argparse
import os
import sys

from eafa import Emotion_Aware_Facial_Animation
from tqdm import tqdm

RAIDROOT = os.environ['RAIDROOT']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model_path', type=str)
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
    T=8,
    n_latent_vec=4,
    normalize_audio=False
)

dataset = args.dataset

if os.path.exists(f'/home/meissen/Datasets/{dataset}/'):
    root_path = f'/home/meissen/Datasets/{dataset}/'
else:
    root_path = RAIDROOT + f'Datasets/{dataset}/'
audio_root = root_path + 'Audio/'
latent_root = root_path + 'Aligned256/'
target_path = f'{root_path}results_own_model_{args.model_path.split("/")[-3]}/'
os.makedirs(target_path, exist_ok=True)

videos = []
with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
    line = f.readline()
    while line:
        videos.append(line.replace('\n', ''))
        line = f.readline()

pbar = tqdm(total=len(videos))
for video in videos:
    latentfile = f"{latent_root}{video}/mean.latent.pt"
    sentence = f"{latent_root}{video}/"
    audiofile = f"{audio_root}{video}.wav"

    pbar.set_description(video)

    # Create video
    max_sec = 30 if dataset == 'AudioDataset' else None
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
