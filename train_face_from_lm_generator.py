import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from my_models import models
from my_models.style_gan_2 import Generator
from PIL import ImageDraw
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from utils import datasets, utils


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
        self.g = Generator(
            1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        for param in self.g.parameters():
            param.requires_grad = False

        # Init global step
        self.global_step = 0

        # Define encoder model
        self.e = models.lmToStyleGANLatent(
            self.g.latent_avg.unsqueeze(0)).train().to(self.device)

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
        self.criterion = nn.MSELoss()

        # Set up tensorboard
        if not self.args.debug and not self.args.test:
            tb_dir = self.args.save_dir
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

    def get_inputs(self, batch):
        landmarks = batch['landmarks'].to(self.device)
        target = batch['latent'].to(self.device)
        b = landmarks.shape[0]
        img_inp = F.interpolate(batch['img'], size=8, mode='nearest').view(b, -1).to(self.device)

        # from torchvision import transforms
        # from torchvision.utils import make_grid
        # t = transforms.ToPILImage('RGB')
        # t(make_grid(img[0].cpu(), normalize=True, range=(-1, 1))).show()
        # img_small = F.interpolate(img, size=8, mode='nearest')
        # t(make_grid(img_small[0].cpu(), normalize=True, range=(-1, 1))).show()
        # 1 / 0
        inp = torch.cat((img_inp, landmarks.view(b, -1)), dim=1)

        return inp, target, landmarks

    def forward(self, inp, evaluate=False):
        if evaluate:
            self.e.eval()
        else:
            self.e.train()
        prediction = self.e(inp)
        return prediction

    def train(self, n_iters, train_loader, val_loader):
        print("Start training")
        val_loss = 0.0

        pbar = tqdm()
        pbar.total = n_iters
        i_iter = 0
        while i_iter < n_iters:
            for batch in train_loader:
                # Unpack batch
                inp, target, lm = self.get_inputs(batch)

                # Update learning rate
                t = self.global_step / n_iters
                self.update_lr(t)

                pred = self.forward(inp, evaluate=False)
                loss = self.criterion(pred, target).mean()

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1
                i_iter += 1
                pbar.update()

                # Update progress bar
                if self.global_step % self.args.update_pbar_every == 0:
                    pbar.set_description('Step {gs} - '
                                         'Train loss {tl:.4f} - '
                                         'Val loss {vl:.4f} - '
                                         'lr {lr:.4f}'.format(
                                             gs=self.global_step,
                                             tl=loss,
                                             vl=val_loss,
                                             lr=self.lr
                                         ))

                if not self.args.debug:
                    if self.global_step % self.args.log_train_every == 0:
                        self.writer.add_scalars(
                            'loss', {'train': loss}, self.global_step)

                    if self.global_step % self.args.log_val_every == 0:
                        val_loss, val_pred, val_target, val_lm = self.eval(val_loader)
                        self.writer.add_scalars(
                            'loss', {'val': val_loss}, self.global_step)

                    if self.global_step % self.args.save_every == 0:
                        self.save()

                    if self.global_step % self.args.save_img_every == 0:
                        if val_target is not None and val_pred is not None:
                            self.save_image(pred, target, lm, name=f'train_sample{self.global_step}.png')
                            self.save_image(
                                val_pred, val_target, val_lm, name=f'val_sample{self.global_step}.png')

                # Break if n_iters is reached and still in epoch
                if i_iter == n_iters:
                    break

        self.save()
        print('Done.')

    def eval(self, val_loader):
        # Validation sample
        batch = next(iter(val_loader))
        inp, target, lm = self.get_inputs(batch)

        with torch.no_grad():
            pred = self.forward(inp, evaluate=True)
            loss = self.criterion(pred, target).mean()

        return loss, pred, target, lm

    def save_image(self, pred, target, lm, name, n_sample=4):

        def draw_landmarks(img, lm):
            img = transforms.ToPILImage('RGB')(img)
            draw = ImageDraw.Draw(img)
            for (x, y) in lm:
                draw.point((x, y), fill='blue')
            img = transforms.ToTensor()(img)
            return img

        # Save train sample
        with torch.no_grad():
            # Generate image
            img = self.g([pred[:n_sample]], input_is_latent=True,
                         noise=self.g.noises)[0].cpu()
            img_target = self.g(
                [target[:n_sample]], input_is_latent=True, noise=self.g.noises)[0].cpu()
            # downsample to 256
            img = utils.downsample_256(img)
            img_target = utils.downsample_256(img_target)
            # Normalize
            img = torch.stack([make_grid(i, normalize=True, range=(-1, 1)) for i in img])
            img_target = torch.stack([make_grid(i, normalize=True, range=(-1, 1)) for i in img_target])
            # Denormalize lm
            lm = (lm + 1.) * 127.5
            # Draw landmarks on img
            img = torch.stack([draw_landmarks(i, l) for i, l in zip(img, lm)])
            img_target = torch.stack([draw_landmarks(i, l) for i, l in zip(img_target, lm)])
        save_tensor = torch.cat((img_target, img), dim=0)
        save_image(
            save_tensor,
            self.args.save_dir + name,
            nrow=min(8, n_sample)
        )

    def test_model(self, val_loader):
        batch = next(iter(val_loader))
        inp, target, lm = self.get_inputs(batch)
        lm = lm[2]
        lm = (lm + 1.) * 127.5

        with torch.no_grad():
            pred = self.forward(inp[2].unsqueeze(0), evaluate=True)
            img = self.g([pred],
                         input_is_latent=True, noise=self.g.noises)[0].cpu()
            img = utils.downsample_256(img)

        img = make_grid(img, normalize=True, range=(-1, 1))
        img = transforms.ToPILImage('RGB')(img)
        draw = ImageDraw.Draw(img)
        for (x, y) in lm:
            draw.point((x, y), fill='blue')
        img.show()


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Flags
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)

    # Hparams
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=0.001)

    # Logging params
    parser.add_argument('--n_iters', type=int, default=3000)
    parser.add_argument('--update_pbar_every', type=int, default=10)
    parser.add_argument('--log_train_every', type=int, default=10)
    parser.add_argument('--log_val_every', type=int, default=100)
    parser.add_argument('--save_img_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str,
                        default='saves/face_from_landmark_generator/')
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

    # Data loading
    train_paths, val_paths = datasets.ravdess_get_paths_actor_split(
        root_path=HOME + "/Datasets/RAVDESS/Aligned256/",
        flat=True,
        shuffled=True,
        validation_split=0.1
    )
    train_ds = datasets.RAVDESSFlatDataset(
        paths=train_paths,
        device=device,
        load_landmarks=True,
        load_latent=True,
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5]
    )
    val_ds = datasets.RAVDESSFlatDataset(
        paths=val_paths,
        device=device,
        load_landmarks=True,
        load_latent=True,
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5]
    )
    val_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    train_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    print(len(train_ds), len(val_ds))

    # Init solver
    solver = solverEncoder(args)

    # Train
    if args.test:
        solver.test_model(val_loader)
    else:
        solver.train(args.n_iters, train_loader, val_loader)
