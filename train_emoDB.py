import argparse
import numpy as np
import os
import random
import torch

from datetime import datetime
from lpips import PerceptualLoss
from my_models.style_gan_2 import Generator
from my_models.models import EmoDBResNet, EmotionDatabase
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import datasets, utils
from torchvision.utils import save_image


HOME = os.path.expanduser('~')


class Solver:
    def __init__(self, args, len_ds):
        super().__init__()

        self.device = args.device
        self.args = args

        self.initial_lr = self.args.lr
        self.lr = self.args.lr
        self.lr_rampdown_length = 0.4
        self.lr_rampup_length = 0.1
        self.reg_weight = 0.01

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

        # Define encoder model and emotional db
        self.e = EmoDBResNet().train().to(self.device)
        self.db = EmotionDatabase(len_ds).to(device)

        if self.args.cont or self.args.test:
            model_path = self.args.model_path
            db_path = self.args.db_path
            self.e.load_state_dict(torch.load(model_path))
            self.db.load_state_dict(torch.load(db_path))
            self.global_step = int(model_path.split(
                '/')[-1].split('.')[0].split('model')[-1])

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.e),
            utils.count_trainable_params(self.e)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(list(self.e.parameters()) + list(self.db.parameters()), lr=self.initial_lr)
        self.criterion = PerceptualLoss(
            model='net-lin', net='vgg').to(self.device)

        # Set up tensorboard
        if self.args.log and not self.args.test:
            self.args.save_dir.split('/')[-1]
            tb_dir = 'tensorboard_runs/emo_db/' + \
                self.args.save_dir.split('/')[-2]
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

        # Create save dir
        os.makedirs(self.args.save_dir + 'models', exist_ok=True)

    def save(self):
        # Saving encoder
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.e.state_dict(), save_path)

        # Saving db
        save_path = f"{self.args.save_dir}models/db{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.db.state_dict(), save_path)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt.param_groups[0]['lr'] = self.lr

    def train(self, data_loaders, n_epochs):
        print("Start training")
        n_iters = n_epochs * len(data_loaders['train'])
        for i_epoch in range(1, n_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, n_epochs))
            print('-' * 10)

            pbar = tqdm(data_loaders['train'])
            for batch in pbar:
                # Unpack batch
                img = batch['x'].view(-1, *sample['x'].shape[-3:]).to(device)
                index = batch['index'].view(-1,)

                # Update learning rate
                t = self.global_step / n_iters
                self.update_lr(t)

                # Encode
                emo_vec = self.db(index)
                latent_offset = self.e(img, emo_vec)
                # Add mean (we only want to compute offset to mean latent)
                latent = latent_offset + self.latent_avg

                # Decode
                img_gen, _ = self.g(
                    [latent], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_gen = utils.downsample_256(img_gen)

                # Compute perceptual loss
                p_loss = self.criterion(img_gen, img).mean()

                # Regularization term
                first = emo_vec[torch.arange(0, 8, 2)]
                second = emo_vec[torch.arange(1, 8, 2)]
                reg_loss = (((first - second) ** 2).mean() * 2) / (emo_vec ** 2).mean()
                reg_loss *= self.reg_weight

                loss = p_loss + reg_loss

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    pbar.set_description('Step [{gs}/{ni}] - '
                                         'p_loss {pl:.4f} - '
                                         'reg_loss {rl:.4f} - '
                                         'lr {lr:.4f}'.format(
                                             gs=self.global_step,
                                             ni=n_iters,
                                             pl=p_loss,
                                             rl=reg_loss,
                                             lr=self.lr
                                         ))
                    if self.args.log:
                        self.writer.add_scalar(
                            'train/Loss', p_loss, self.global_step)

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
                    print("")

        self.save()
        print('Done.')

    def test_model(self, ds, n_samples=8):
        """
        Example frames:
        Super happy: Actor_01/01-01-03-02-01-01-01/108.png
        Super angry: Actor_01/01-01-05-02-01-01-01/096.png
        Super sad: Actor_01/01-01-04-02-01-01-01/108.png
        """
        iters = max(n_samples // self.args.batch_size, 1)
        self.e.eval()
        imgs = []
        imgs_gen = []
        for i in range(iters):
            sample = next(iter(data_loaders['val']))
            img = sample['x'].view(-1, *sample['x'].shape[-3:]).to(self.device)
            index = sample['index'].view(-1,)

            with torch.no_grad():
                emo_vec = self.db(index)
                latent_offset = self.e(img, emo_vec)
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

    # Random seeds
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str,
                        default='saves/emo_db/')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--db_path', type=str, default=None)
    args = parser.parse_args()

    if args.cont or args.test:
        assert args.model_path is not None
        assert args.db_path is not None

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
    train_paths, _ = datasets.ravdess_get_paths(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        validation_split=0.0,
        flat=True,
        actors=[1],
        emotions=['neutral', 'happy']
    )
    train_ds = datasets.RAVDESSEmoDBFlatDataset(
        paths=train_paths,
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    data_loaders, _ = datasets.get_data_loaders(
        train_ds, None, args.batch_size, use_cuda=True
    )
    sample = next(iter(data_loaders['train']))

    # Init solver
    solver = Solver(args, len(train_ds))

    # Train
    if args.test:
        solver.test_model(data_loaders, n_samples=8)
    else:
        solver.train(data_loaders, args.n_epochs)
