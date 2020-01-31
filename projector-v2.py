import argparse
import glob
import os
import numpy as np
import sys
import torch

from tqdm import tqdm
from lpips import PerceptualLoss
from my_models.style_gan_2 import Generator
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class Projector:
    def __init__(self,
                 g,
                 num_steps=1000,
                 initial_learning_rate=0.1,
                 initial_noise_factor=0.05,
                 verbose=True,
                 initial_latent=None,
                 ):

        self.num_steps = num_steps
        self.n_mean_latent = 10000
        self.initial_lr = initial_learning_rate
        self.lr = initial_learning_rate
        self.initial_noise_factor = initial_noise_factor
        self.lr_rampdown_length = 0.25
        self.lr_rampup_length = 0.05
        self.noise_ramp_length = 0.75
        self.regularize_noise_weight = 1e5
        self.verbose = verbose

        self.latent_expr = None
        self.lpips = None
        self.target_images = None
        self.loss_appearance = None
        self.loss_expression = None
        self.loss = None
        self.cur_step = None

        self.g_ema = g
        self.device = next(g.parameters()).device

        # Find latent stats
        self._info(
            ('Finding W midpoint and stddev using %d samples...' % self.n_mean_latent))
        torch.manual_seed(123)
        with torch.no_grad():
            noise_sample = torch.randn(
                self.n_mean_latent, 512, device=self.device)
            latent_out = self.g_ema.style(noise_sample)

        self.latent_mean = latent_out.mean(0)
        self.latent_std = (
            (latent_out - self.latent_mean).pow(2).sum() / self.n_mean_latent) ** 0.5
        self._info('std = {}'.format(self.latent_std))
        # self.latent_mean = (torch.randn(512, device=self.device) * 0.1539) + 0.097

        if initial_latent is None:
            self.latent_in = self.latent_mean.detach().clone().unsqueeze(0)
            self.latent_in = self.latent_in.repeat(self.g_ema.n_latent, 1)
        else:
            self.latent_in = initial_latent

        self.latent_appearance = torch.cat((self.latent_in[:3], self.latent_in[5:]), dim=0)
        self.latent_expression = self.latent_in[3:5]
        self.latent_appearance.requires_grad = True
        self.latent_expression.requires_grad = True

        # Find noise inputs.
        self.noises = [noise.to(self.device) for noise in g.noises]

        # Init optimizer
        self.opt_appearance = torch.optim.Adam(
            [self.latent_appearance], lr=self.initial_lr)
        self.opt_expression = torch.optim.Adam(
            [self.latent_expression], lr=self.initial_lr)

        # Init loss function
        self.lpips = PerceptualLoss(model='net-lin', net='vgg').to(self.device)

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt_appearance.param_groups[0]['lr'] = self.lr
        self.opt_expression.param_groups[0]['lr'] = self.lr

    def noise_regularization(self):
        reg_loss = 0.0
        for noise in self.noises:
            size = noise.shape[2]
            while True:
                reg_loss += (noise * noise.roll(1, dims=3)).mean().pow(2) + \
                    (noise * noise.roll(1, dims=2)).mean().pow(2)
                if size <= 8:
                    break  # Small enough already
                noise = noise.reshape(
                    [1, 1, size // 2, 2, size // 2, 2])  # Downscale
                noise = noise.mean(dim=[3, 5])
                size = size // 2
        return reg_loss

    def normalize_noise(self):
        for noise in self.noises:
            mean = noise.mean()
            std = noise.std()
            noise.data.add_(-mean).div_(std)

    def prepare_input(self, target_image):
        if len(target_image.shape) == 3:
            target_image = target_image.unsqueeze(0)
        if target_image.shape[2] > 256:
            target_image = self.downsample_img(target_image)
        return target_image

    def downsample_img(self, img):
        b, c, h, w = img.shape
        factor = h // 256
        img = img.reshape(b, c, h // factor, factor, w // factor, factor)
        img = img.mean([3, 5])
        return img

    def run(self, appearance, expression, num_steps):
        self.num_steps = num_steps
        self.appearance = self.prepare_input(appearance)
        self.expression = self.prepare_input(expression)

        self._info('Running...')
        pbar = tqdm(range(self.num_steps))
        for i_step in pbar:
            self.cur_step = i_step
            self.step()
            pbar.set_description(
                (f'loss: {self.loss.item():.4f}; lr: {self.lr:.4f}'))

    def step(self):
        # Hyperparameters
        t = self.cur_step / self.num_steps

        # Build latent back together
        self.latent_in = torch.cat(
            (self.latent_appearance[:3], self.latent_expression, self.latent_appearance[3:]),
            dim=0
        )

        # Add noise to dlatents
        noise_strength = self.latent_std * self.initial_noise_factor * \
            max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        latent_noise = (torch.randn_like(self.latent_in)
                        * noise_strength).to(self.device)
        self.latent_expr = self.latent_in + latent_noise

        # Update learning rate
        self.update_lr(t)

        # Train
        self.img_gen, _ = self.g_ema(
            [self.latent_expr], input_is_latent=True, noise=self.noises)
        # Downsample to 256 x 256
        self.img_gen = self.downsample_img(self.img_gen)

        # Compute perceptual loss
        self.loss_appearance = self.lpips(self.img_gen, self.appearance).sum()
        self.loss_expression = self.lpips(self.img_gen, self.expression).sum()

        self.loss = self.loss_appearance + self.loss_expression

        # Update params
        self.opt_appearance.zero_grad()
        self.loss_appearance.backward(retain_graph=True)
        self.opt_appearance.step()

        self.opt_expression.zero_grad()
        self.loss_expression.backward()
        self.opt_expression.step()

    def get_images(self):
        self.latent_in = torch.cat(
            (self.latent_appearance[:3], self.latent_expression, self.latent_appearance[3:]),
            dim=0
        )
        imgs, _ = self.g_ema(
            [self.latent_in], input_is_latent=True, noise=self.noises)
        return imgs

    def get_latents(self):
        self.latent_in = torch.cat(
            (self.latent_appearance[:3], self.latent_expression, self.latent_appearance[3:]),
            dim=0
        )
        return self.latent_in.detach()


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('appearance', type=str)
    parser.add_argument('expression', type=str)
    parser.add_argument('--target_dir', type=str,
                        default='saves/fb_expression_transfer/')
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    g = Generator(1024, 512, 8, pretrained=True).to(device).train()
    for param in g.parameters():
        param.requires_grad = False

    proj = Projector(g)

    # Load image
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    appearance = t(Image.open(args.appearance)).to(device)
    expression = t(Image.open(args.expression)).to(device)

    # Get names
    appearance_name = args.appearance.split('/')[-1].split('.')[0]
    expression_name = args.expression.split('/')[-1].split('.')[0]
    print("Projecting for appearance {} and expression {}".format(
        appearance_name, expression_name
    ))

    # Run projector
    proj.run(appearance, expression, 1000)

    # Collect results
    generated = proj.get_images()
    latents = proj.get_latents()

    # Save results
    save_str = 'saves/projected_images/' + \
        f"{appearance_name}-{expression_name}-g"
    os.makedirs('saves/projected_images/', exist_ok=True)
    print('Saving {}'.format(save_str + '.png'))
    save_image(generated, save_str + '.png', normalize=True)
    torch.save(latents.detach().cpu(), save_str + '.pt')
