import argparse
import numpy as np
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

        self.initial_lr = self.args.lr
        self.lr = self.args.lr
        self.lr_rampdown_length = 0.3
        self.lr_rampup_length = 0.1

        # Load generator
        self.g = Generator(1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        for param in self.g.parameters():
            param.requires_grad = False
        self.latent_avg = self.g.latent_avg.repeat(18, 1).unsqueeze(0).to(self.device)

        # Init global step
        self.global_step = 0

        # Define encoder model
        self.e = resnetEncoder().train().to(self.device)

        if self.args.cont or self.args.test:
            path = self.args.model_path
            self.e.load_state_dict(torch.load(path))
            self.global_step = int(path.split(
                '/')[-1].split('.')[0].split('model')[-1])

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.e),
            utils.count_trainable_params(self.e)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(self.e.parameters(), lr=self.initial_lr)
        self.criterion = PerceptualLoss(model='net-lin', net='vgg').to(self.device)

        # Set up tensorboard
        if self.args.log and not self.args.test:
            self.args.save_dir.split('/')[-1]
            tb_dir = 'tensorboard_runs/encode_stylegan/' + \
                self.args.save_dir.split('/')[-2]
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

        # Create save dir
        os.makedirs(self.args.save_dir + 'models', exist_ok=True)

    def save(self):
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.e.state_dict(), save_path)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt.param_groups[0]['lr'] = self.lr

    def train(self, data_loaders, n_epochs):
        print("Start training")
        n_iters = n_epochs * len(data_loaders['train'])
        val_loss = 0.
        for i_epoch in range(n_epochs):
            print('Epoch {}/{}'.format(i_epoch, n_epochs))
            print('-' * 10)

            pbar = tqdm(data_loaders['train'])
            for batch in pbar:
                # Unpack batch
                img = batch['x'].to(device)

                # Update learning rate
                t = self.global_step / n_iters
                self.update_lr(t)

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
                    pbar.set_description('Step {gs} - '
                                         'train loss {tl:.4f} - '
                                         'val loss {vl:.4f} - '
                                         'lr {lr:.4f}'.format(
                                             gs=self.global_step,
                                             tl=loss,
                                             vl=val_loss,
                                             lr=self.lr
                                         ))
                    if self.args.log:
                        self.writer.add_scalar('train/Loss', loss, self.global_step)

                if self.global_step % self.args.save_every == 0:
                    self.save()

                if self.global_step % self.args.eval_every == 0:
                    # Save train sample
                    save_tensor = torch.cat(
                        (img.detach(), img_gen.detach().clamp(-1., 1.)), dim=0)
                    save_image(
                        save_tensor,
                        f'{self.args.save_dir}train_gen_{self.global_step}.png',
                        normalize=True,
                        nrow=min(8, self.args.batch_size)
                    )

                    # Eval one batch
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
            (img_val.detach(), img_val_gen.detach().clamp(-1., 1.)), dim=0)
        save_image(
            save_tensor,
            f'{self.args.save_dir}val_gen_{self.global_step}.png',
            normalize=True,
            nrow=min(8, self.args.batch_size)
        )

        # Set encoder back to train
        self.e.train()

        return val_loss

    def test_model(self, data_loaders, n_samples=8):
        iters = max(n_samples // self.args.batch_size, 1)
        self.e.eval()
        imgs = []
        imgs_gen = []
        for i in range(iters):
            sample = next(iter(data_loaders['val']))
            img = sample['x'].to(self.device)

            with torch.no_grad():
                latent_offset = self.e(img)
                # Add mean (we only want to compute offset to mean latent)
                latent = latent_offset + self.latent_avg

                # Decode
                img_gen, _ = self.g(
                    [latent], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_gen = utils.downsample_256(img_gen)

            imgs.append(img)
            imgs_gen.append(img_gen)
        imgs = torch.cat(imgs, dim=0)
        imgs_gen = torch.cat(imgs_gen, dim=0)

        img_tensor = torch.cat((imgs, imgs_gen), dim=0)

        save_image(
            img_tensor,
            f'{self.args.save_dir}test_model.png',
            normalize=True,
            nrow=n_samples
        )
        self.e.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='saves/encode_stylegan/')
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

    # Load data
    ds = datasets.RAVDESSFlatDataset(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
        actors=[1],
    )
    data_loaders, dataset_sizes = datasets.get_data_loaders(
        ds, validation_split=0.8, batch_size=args.batch_size, use_cuda=True)

    # Init solver
    solver = solverEncoder(args)

    # Train
    if args.test:
        solver.test_model(data_loaders, n_samples=8)
    else:
        solver.train(data_loaders, args.n_epochs)
