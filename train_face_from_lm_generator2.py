import argparse
import os
import torch
import torch.nn.functional as F

from datetime import datetime
from my_models import discriminators, generators
from my_models.model_utils import weights_init
from utils import datasets, utils
from lpips import PerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

HOME = os.path.expanduser('~')


class Solver(object):
    def __init__(self, args):
        # General
        self.args = args
        self.device = args.device
        self.global_step = 0

        print("Training on {}".format(self.device))

        # Models
        self.init_models()

        # Optimizers
        self.init_optimizers()

        # Loss Functions
        self.criterionGAN = utils.GANLoss(args.gan_mode, self.device,
                                          flip_p=args.flip_prob,
                                          noisy_labels=args.noisy_labels,
                                          label_range_real=args.label_range_real,
                                          label_range_fake=args.label_range_fake)
        self.criterionPix = torch.nn.L1Loss()
        self.lpips = PerceptualLoss(model='net-lin', net='vgg').to(self.device)

        # Init tensorboard
        if not self.args.debug:
            tb_dir = self.args.save_dir
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

            # Create save dir
            os.makedirs(self.args.save_dir + 'models', exist_ok=True)

    def init_models(self):
        # Init generator
        self.generator = generators.AdaINGenerator128(
            3 * 8 * 8 + 68 * 2).to(self.device)
        self.generator.apply(weights_init)

        print("Generator params {} (trainable {})".format(
            utils.count_params(self.generator),
            utils.count_trainable_params(self.generator)
        ))

        # Init discriminator
        self.discriminator = discriminators.SimpleDiscriminator128().to(self.device)
        self.discriminator.apply(weights_init)

        print("Discriminator params {} (trainable {})".format(
            utils.count_params(self.discriminator),
            utils.count_trainable_params(self.discriminator)
        ))

    def init_optimizers(self):
        params_G = list(self.generator.parameters())
        params_D = list(self.discriminator.parameters())

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            params_G,
            lr=self.args.lr_G,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            params_D,
            lr=self.args.lr_D,
            betas=(0.5, 0.999)
        )

    def save(self):
        # Generator
        generator_path = self.args.save_dir + \
            f'models/generator{self.global_step}.pt'
        print(f'Saving generator to {generator_path}')
        torch.save(self.generator.state_dict(), generator_path)

        # Discriminator
        disciminator_path = self.args.save_dir + \
            f'models/disciminator_path{self.global_step}.pt'
        print(f'Saving disciminator to {disciminator_path}')
        torch.save(self.discriminator.state_dict(), disciminator_path)

    def log_tensorboard(self, d_losses, g_losses):
        for name, loss in d_losses.items():
            self.writer.add_scalars(
                'discriminator', {name: loss}, self.global_step)

        for name, loss in g_losses.items():
            self.writer.add_scalars(
                'generator', {name: loss}, self.global_step)

    def unpack_batch(self, batch):
        imgs = batch['img'].to(self.device)
        b = imgs.shape[0]

        reals = imgs

        lm = batch['landmarks'].view(b, -1).to(self.device)
        imgs_small = F.interpolate(imgs, (8, 8)).view(b, -1)
        fake_inp = torch.cat((imgs_small, lm), dim=1)

        return reals, fake_inp

    def backward_D(self, reals, fakes):
        d_losses = {}
        d_losses['total'] = 0.
        if self.args.lambda_GAN:
            if self.args.gan_mode == 'wgan':
                # clamp parameters to a cube
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # All real batch
            pred_real = self.discriminator(reals)
            d_loss_real = self.criterionGAN(
                pred_real, True, for_discriminator=True)
            d_losses['real'] = d_loss_real.item()

            # All fake batch
            pred_fake = self.discriminator(fakes)
            d_loss_fake = self.criterionGAN(
                pred_fake, False, for_discriminator=True)
            d_losses['fake'] = d_loss_fake.item()

            # Combine losses
            if self.args.gan_mode == 'wgan':
                d_loss_total = d_loss_real - d_loss_fake
            else:
                d_loss_total = d_loss_real + d_loss_fake
            d_losses['total'] = d_loss_total.item()

            # Backward
            d_loss_real.backward(retain_graph=True)
            d_loss_fake.backward(retain_graph=True)

        return d_losses

    def backward_G(self, reals, fakes):
        # GAN loss
        g_losses = {}
        g_loss_total = 0.

        if self.args.lambda_GAN:
            pred_fake = self.discriminator(fakes)
            g_loss_gan = self.criterionGAN(
                pred_fake, True, for_discriminator=False) * self.args.lambda_GAN
            g_loss_total += g_loss_gan
            g_losses['gan'] = g_loss_gan.item()

        # Pixelwise loss
        if self.args.lambda_pixel:
            g_loss_pixel = self.criterionPix(
                fakes, reals) * self.args.lambda_pixel
            g_loss_total += g_loss_pixel
            g_losses['pixel'] = g_loss_pixel.item()

        # VGG loss
        if self.args.lambda_lpips:
            g_loss_lpips = self.lpips(
                fakes, reals).mean() * self.args.lambda_lpips
            g_loss_total += g_loss_lpips
            g_losses['lpips'] = g_loss_lpips.item()

        # Backward
        g_loss_total.backward()

        g_losses['total'] = g_loss_total
        return g_losses

    def train(self, data_loaders, n_iters):

        print("Starting training")

        pbar = tqdm(total=n_iters)
        i_iter = 0
        while i_iter < n_iters:
            for batch in data_loaders['train']:

                # Increment step counter
                self.global_step += 1
                i_iter += 1
                pbar.update()

                # Set inputs
                reals, fake_inp = self.unpack_batch(batch)

                # Forward
                fakes = self.generator(fake_inp)

                # (1) Train discriminator
                self.optimizer_D.zero_grad()
                d_losses = self.backward_D(reals, fakes)
                self.optimizer_D.step()

                # (2) Train Generator
                self.optimizer_G.zero_grad()
                g_losses = self.backward_G(reals, fakes)
                self.optimizer_G.step()

                # Update progress bar
                pbar.set_description('Step {gs} - '
                                     'G loss {gl:.4f} - '
                                     'D loss {dl:.4f} - '.format(
                                         gs=self.global_step,
                                         gl=g_losses['total'],
                                         dl=d_losses['total'],
                                     ))

                if not self.args.debug:
                    # Tensorboar logging
                    self.log_tensorboard(d_losses, g_losses)

                    if self.global_step % self.args.save_every == 0:
                        # Save model
                        self.save()

                    if self.global_step % self.args.eval_every == 0:
                        self.eval(data_loaders)

                # Break if n_iters is reached in middle of epoch
                if i_iter == n_iters:
                    break

        if not self.args.debug:
            self.save()

    def eval(self, data_loaders, n_samples=4):
        print("Evaluating generator")

        # Real images vs fake images
        batch = next(iter(data_loaders['train']))
        reals, fake_inp = self.unpack_batch(batch)
        reals = reals[:n_samples]
        fake_inp = fake_inp[:n_samples]
        with torch.no_grad():
            fakes = self.generator(fake_inp)

        reals = torch.stack(
            [make_grid(r, normalize=True, range=(-1, 1)) for r in reals])
        fakes = torch.stack(
            [make_grid(f, normalize=True, range=(-1, 1)) for f in fakes])

        real_img = make_grid(reals, normalize=False)
        fake_img = make_grid(fakes, normalize=False)

        # Cat real and fake together
        imgs = torch.stack((real_img, fake_img), 0)
        save_image(
            imgs,
            self.args.save_dir + f'sample_{self.global_step}.png',
            normalize=False,
            nrow=n_samples
        )


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Flags
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)

    # Logging params
    parser.add_argument('--n_iters', type=int, default=150000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--save_dir', type=str,
                        default='saves/face_from_landmark_generator/')

    # Hparams
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr_D', type=float, default=0.0002)
    parser.add_argument('--lr_G', type=float, default=0.0002)
    parser.add_argument('--lambda_GAN', type=float, default=1.)  # 1.
    parser.add_argument('--lambda_pixel', type=float, default=0.)  # 100.
    parser.add_argument('--lambda_lpips', type=float, default=100.)  # 0.

    # GAN hacks
    # 'vanilla' | 'lsgan' | 'wgan'
    parser.add_argument('--gan_mode', type=str, default='vanilla')
    parser.add_argument('--noisy_labels', type=bool, default=True)
    parser.add_argument('--label_range_real', type=tuple, default=(0.9, 1.0))
    parser.add_argument('--label_range_fake', type=tuple, default=(0.0, 0.2))
    parser.add_argument('--grad_clip_val', type=float, default=0.0)
    parser.add_argument('--flip_prob', type=float, default=0.05)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    if args.cont or args.test:
        args.save_dir = '/'.join(args.model_path.split('/')[:-2]) + '/'

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
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5],
        image_size=args.image_size
    )
    val_ds = datasets.RAVDESSFlatDataset(
        paths=val_paths,
        device=device,
        load_landmarks=True,
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5],
        image_size=args.image_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize solver
    solver = Solver(args)

    # Train
    solver.train(data_loaders, args.n_iters)
