import argparse
import copy
import os
import torch

from datetime import datetime
from lpips import PerceptualLoss
from my_models.style_gan_2 import Generator
from my_models.models import resnetEncoder
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
        self.g = Generator(1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        for param in self.g.parameters():
            param.requires_grad = False
        self.latent_avg = self.g.latent_avg.repeat(18, 1).unsqueeze(0).to(self.device)

        # Define encoder model
        self.e = resnetEncoder().to(self.device)

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.e),
            utils.count_trainable_params(self.e)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(self.e.parameters(), lr=args.lr)
        self.criterion = PerceptualLoss(model='net-lin', net='vgg').to(self.device)

        # Init global step
        self.global_step = 0

        # Set up tensorboard
        if self.args.log:
            tb_dir = 'tensorboard_runs/encode_stylegan/' + \
                datetime.now().strftime("%Y%m%d-%H%M%S")
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
        val_loss = 0.
        for i_epoch in range(n_epochs):
            print('Epoch {}/{}'.format(i_epoch, n_epochs))
            print('-' * 10)

            pbar = tqdm(data_loaders['train'])
            for batch in pbar:
                # Unpack batch
                img = batch['x'].to(device)

                # Encode
                latent_offset = self.e(img)
                # Add mean (we only want to compute offset to mean latent)
                latent = latent_offset + self.latent_avg

                # Decode
                img_gen, _ = self.g([latent], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_gen = utils.downsample_256(img_gen)

                # Compute perceptual loss
                loss = self.criterion(img_gen, img).mean()

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    pbar.set_description(f'train loss: {loss:.4f}; val loss: {val_loss:.4f}')
                    if self.args.log:
                        self.writer.add_scalar('train/Loss', loss, self.global_step)

                if self.global_step % self.args.save_every == 0:
                    self.save()

                if self.global_step % self.args.eval_every == 0:
                    # Save train sample
                    save_tensor = torch.cat(
                        (img.detach(), img_gen.detach()), dim=0)
                    save_image(
                        save_tensor,
                        f'{self.args.save_dir}train_gen_{self.global_step}.png',
                        normalize=True
                    )

                    # Eval one batch
                    if data_loaders['val'] is not None:
                        val_loss = self.eval(data_loaders['val'])

        self.save()
        print('Done.')

    def eval(self, val_loader):
        # Set encoder to eval
        self.e.eval()

        # Get random validation batch
        val_batch = next(iter(val_loader))

        # Unpack batch
        img_val = val_batch['x'].to(device)

        # Encode
        with torch.no_grad():
            latent_offset = self.e(img_val)
            # Add mean (we only want to compute offset to mean latent)
            latent = latent_offset + self.latent_avg

            # Decode
            img_val_gen, _ = self.g(
                [latent], input_is_latent=True, noise=self.g.noises)

            # Downsample to 256 x 256
            img_val_gen = utils.downsample_256(img_val_gen)

            # Compute perceptual loss
            val_loss = self.criterion(img_val_gen, img_val).mean()

            if self.args.log:
                self.writer.add_scalar(
                    'val/Loss', val_loss, self.global_step)

        # Save val sample
        save_tensor = torch.cat(
            (img_val.detach(), img_val_gen.detach()), dim=0)
        save_image(
            save_tensor,
            f'{self.args.save_dir}val_gen_{self.global_step}.png',
            normalize=True
        )

        # Set encoder back to train
        self.e.train()

        return val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='saves/encode_stylegan/')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += datetime.now().strftime("%Y%m%d-%H%M%S/")

    print("Saving run to {}".format(args.save_dir))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Load data
    ds = datasets.RAVDESSFlatDataset(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    data_loaders, dataset_sizes = datasets.get_data_loaders(
        ds, validation_split=0.8, batch_size=args.batch_size, use_cuda=True)

    # Init solver
    solver = solverEncoder(args)

    # Train
    solver.train(data_loaders, args.n_epochs)
