import os
import sys

from eafa import Emotion_Aware_Facial_Animation
from tqdm import tqdm

RAIDROOT = os.environ['RAIDROOT']

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
audio_root = root_path + 'Audio/'
latent_root = root_path + 'Aligned256/'
target_path = sys.argv[2]
os.makedirs(target_path, exist_ok=True)

videos = []
with open(root_path + 'grid_videos.txt', 'r') as f:
    line = f.readline()
    while line:
        videos.append(line.replace('\n', ''))
        line = f.readline()

for video in tqdm(videos):
    latentfile = f"{latent_root}{video}/mean.latent.pt"
    sentence = f"{latent_root}{video}/"
    audiofile = f"{audio_root}{video}.wav"

    # Create video
    vid = model(test_latent=latentfile, test_sentence_path=sentence)

    # Visualize
    # from torchvision import transforms
    # transforms.ToPILImage()(vid[0]).show()
    # 1 / 0

    # Save video
    model.save_video(vid, audiofile, f"{target_path}{video}.avi")
