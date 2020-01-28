import argparse
import numpy as np
import os
import random
import torch

from datetime import datetime
from lpips import PerceptualLoss
from my_models.style_gan_2 import Generator
from my_models import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import datasets, utils
from torchvision.utils import save_image


HOME = os.path.expanduser('~')


class solverEncoder:
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
        self.e = models.neutralToXResNet().to(self.device).train()

        if self.args.cont or self.args.test:
            path = self.args.model_path
            self.e.load_state_dict(torch.load(path))
            self.global_step = int(path.split('/')[-1].split('.')[0].split('model')[-1])

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.e),
            utils.count_trainable_params(self.e)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(self.e.parameters(), lr=args.lr)
        self.criterion = PerceptualLoss(
            model='net-lin', net='vgg').to(self.device)

        # Set up tensorboard
        if self.args.log and not self.args.test:
            self.args.save_dir.split('/')[-1]
            tb_dir = 'tensorboard_runs/neutral_to_x/' + \
                self.args.save_dir.split('/')[-2]
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

        # Create save dir
        os.makedirs(self.args.save_dir + 'models', exist_ok=True)

    def save(self):
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.e.state_dict(), save_path)

    def train(self, data_loaders, n_epochs):
        print("Start training")
        pbar = tqdm(range(n_epochs))
        for i_epoch in pbar:
            for _, batch in enumerate(data_loaders['train']):
                # Unpack batch
                img = batch['src'].to(device)
                target = batch['target'].to(device)
                target_happy_score = batch['target_happy_score'].to(device)

                # Encode
                latent_offset = self.e(img, target_happy_score)
                # Add mean (we only want to compute offset to mean latent)
                latent = latent_offset + self.latent_avg

                # Decode
                img_gen, _ = self.g(
                    [latent], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_gen = utils.downsample_256(img_gen)

                # Compute perceptual loss
                loss = self.criterion(img_gen, target).mean()

                # loss = p_loss + 0.05 * F.mse_loss(img_gen, target)

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    pbar.set_description(
                        f'train loss: {loss:.4f}')
                    if self.args.log:
                        self.writer.add_scalar(
                            'train/Loss', loss, self.global_step)

                if self.global_step % self.args.save_every == 0:
                    self.save()

                if self.global_step % self.args.eval_every == 0:
                    # Save train sample
                    save_tensor = torch.cat(
                        (img.detach(), target.detach(), img_gen.detach().clamp(-1., 1.)), dim=0)
                    save_image(
                        save_tensor,
                        f'{self.args.save_dir}train_gen_{self.global_step}.png',
                        normalize=True,
                        nrow=min(8, self.args.batch_size)
                    )

        self.save()
        print('Done.')

    def test_model(self, ds, n_img=8):
        sample = next(iter(ds))
        img = sample['src'].unsqueeze(0).to(self.device)
        scores = torch.tensor(np.linspace(0., 1., n_img),
                              dtype=torch.float32, device=device)

        imgs = []
        self.e.eval()
        for score in scores:
            # Encode
            print(f"Score: {score.item():.4f}")
            with torch.no_grad():
                latent_offset = self.e(img, score.view((1, -1)))
                # Add mean (we only want to compute offset to mean latent)
                latent = latent_offset + self.latent_avg

                # Decode
                img_gen, _ = self.g(
                    [latent], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_gen = utils.downsample_256(img_gen)

            imgs.append(img_gen)
        imgs = torch.cat(imgs, dim=0)

        save_image(
            imgs,
            f'{self.args.save_dir}eval_scores.png',
            normalize=True,
        )
        self.e.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='saves/neutral_to_x/')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    if args.cont or args.test:
        assert args.model_path is not None

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    if args.cont or args.test:
        args.save_dir = '/'.join(args.model_path.split('/')[:-2]) + '/'

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
    solver = solverEncoder(args)

    # Train
    if args.test:
        solver.test_model(ds)
    else:
        solver.train(data_loaders, args.n_epochs)
