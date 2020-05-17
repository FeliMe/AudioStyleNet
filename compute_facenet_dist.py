import numpy as np
import os
import sys
import torch

from eafa import Emotion_Aware_Facial_Animation
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
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


def compute_metric(prediction, static_image, metric_fn):
    metric_arr = []
    for frame_pred in prediction:
        img_pred = Image.fromarray(frame_pred)

        # Visualize
        # img_pred.show()
        # static_image.show()
        # 1 / 0

        metric = metric_fn(img_pred, static_image)
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

    metric_name = 'face_net_dist'
    metric_fn = FaceNetDist(image_size=109)

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
        # print(f"Image {imagefile} - Audio {audiofile})

        static_image = image_from_latent(latentfile, model)
        static_image = transforms.ToPILImage()(static_image)

        # Create video
        vid = model(test_latent=latentfile, test_sentence_path=sentence)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        # Compute metric
        metric = compute_metric(vid, static_image, metric_fn)
        metric_mean += metric.mean()
        pbar.update()
        pbar.set_description(f"{metric_name}: {metric.mean():.4f}")

    print(f"mean {metric_name}: {metric_mean / len(videos):.4f}")
    print(f"prediction was {root_path}")
