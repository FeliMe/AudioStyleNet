import argparse
import numpy as np
import os
import random
import torch

from datetime import datetime
from my_models import models
from my_models.style_gan_2 import Generator, CycleD
from utils import datasets


HOME = os.path.expanduser('~')


class solverXToYCycle:
    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.args = args

        # Load generator
        self.g = Generator(
            1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        for param in self.g.parameters():
            param.requires_grad = False
        self.latent_avg = self.g.latent_avg.repeat(
            18, 1).unsqueeze(0).to(self.device)

        # Init global step
        self.global_step = 0

        # Define encoder model
        self.e_x = models.neutralToXResNet().to(self.device)
        self.e_y = models.neutralToXResNet().to(self.device)

        # Define discriminator
        self.d_x = CycleD(256).to(self.device)
        self.d_y = CycleD(256).to(self.device)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str,
                        default='saves/x_to_y_cycle/')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    print("Saving run to {}".format(args.save_dir))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Random seeds
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    ds = datasets.RAVDESSPseudoPairDataset(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
        actors=[1],
    )
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )
    }

    # Init solver
    solver = solverXToYCycle(args)

    # Train
    solver.train(data_loaders, args.n_epochs)
