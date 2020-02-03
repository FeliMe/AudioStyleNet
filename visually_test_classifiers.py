import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms

from my_models import models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import datasets

HOME = os.path.expanduser('~')


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emotion', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Select model
    if args.model == 'fer':
        model = models.FERClassifier(emotions=[args.emotion])
    elif args.model == 'ravdess':
        model = models.EmotionClassifier(emotions=[args.emotion])
    else:
        raise NotImplementedError('Wrong model')
    model = model.eval().to(device)

    # Transforms
    t = transforms.ToTensor()

    # Init dataset
    paths, _, _ = datasets.get_paths(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        flat=True,
        emotions=[args.emotion],
        actors=[1]
    )
    ds = datasets.RAVDESSFlatDataset(
        paths=paths,
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256
    )
    data_loader = DataLoader(ds, batch_size=1)

    # Get scores for ds
    scores = []
    paths = []
    for sample in data_loader:
        img = sample['x'].to(device)
        path = sample['paths'][0]
        score = model(img)[0].cpu()
        scores.append(score)
        paths.append(path)

    paths = [p for _, p in sorted(zip(scores, paths))]
    scores = sorted(scores)

    inds = np.linspace(0, len(scores) - 1, num=8, dtype=int)
    selected_paths = []
    for i in inds:
        selected_paths.append(paths[i])
    imgs = torch.stack([t(Image.open(p)) for p in selected_paths], dim=0)

    title = '{m}_{e}_from_{low:.4f}_to_{high:.4f}.png'.format(
        m=args.model,
        e=args.emotion,
        low=scores[0][0],
        high=scores[-1][0]
    )

    plt.plot(scores)

    save_dir = 'saves/test_classifiers/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + '{}_{}_distribution.png'.format(args.model, args.emotion))
    save_image(imgs, save_dir + title, normalize=True)
