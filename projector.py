import os
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms
from lpips import PerceptualLoss
from tqdm import tqdm


class Projector:
    def __init__(self,
                 num_steps=1000,
                 initial_learning_rate=0.1,
                 initial_noise_factor=0.05,
                 verbose=True
                 ):

        self.num_steps = num_steps
        self.n_mean_latent = 10000
        self.initial_lr = initial_learning_rate
        self.initial_noise_factor = initial_noise_factor
        self.lr_rampdown_length = 0.25
        self.lr_rampup_length = 0.05
        self.noise_ramp_length = 0.75
        self.regularize_noise_weight = 1e5
        self.verbose = verbose

        self.g_ema = None
        self.device = None
        self.latent_mean = None
        self.latent_std = None
        self.noises = None
        self.latent_in = None
        self.latent_expr = None
        self.lpips = None
        self.target_images = None
        self.imag_gen = None
        self.loss = None
        self.lr = None
        self.opt = None
        self.cur_step = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt.param_groups[0]['lr'] = self.lr

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

    def prepare_input(self, target_images):
        if len(target_images.shape) == 3:
            target_images = target_images.unsqueeze(0)
        self.target_images = target_images

    def downsample_img(self, img):
        b, c, h, w = img.shape
        factor = h // 256
        img = img.reshape(b, c, h // factor, factor, w // factor, factor)
        img = img.mean([3, 5])
        return img

    def set_network(self, g, minibatch_size=1):
        self.g_ema = g.g
        self.device = next(g.parameters()).device

        # Find latent stats
        self._info(('Finding W midpoint and stddev using %d samples...' % self.n_mean_latent))
        torch.manual_seed(123)
        with torch.no_grad():
            noise_sample = torch.randn(self.n_mean_latent, 512, device=self.device)
            latent_out = self.g_ema.style(noise_sample)

            self.latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - self.latent_mean).pow(2).sum() /
                               self.n_mean_latent) ** 0.5
            self._info('std = {}'.format(self.latent_std))

        self.latent_in = self.latent_mean.detach().clone().unsqueeze(0)
        self.latent_in = self.latent_in.repeat(self.g_ema.n_latent, 1)
        self.latent_in.requires_grad = True

        # Find noise inputs.
        self.noises = self.g_ema.make_noise()
        for noise in self.noises:
            noise.requires_grad = True

        # Init optimizer
        self.opt = torch.optim.Adam([self.latent_in] + self.noises, lr=self.initial_lr)
        # self.opt = torch.optim.Adam([self.latent_in], lr=self.initial_lr)

        # Init loss function
        self.lpips = PerceptualLoss(model='net-lin', net='vgg').to(self.device)

    def run(self, target_images):
        self.prepare_input(target_images)

        self._info('Running...')
        pbar = tqdm(range(self.num_steps))
        for i_step in pbar:
            self.cur_step = i_step
            self.step()
            pbar.set_description((f'loss: {self.loss.item():.4f}; lr: {self.lr:.4f}'))

        # Collect results
        return self.get_images()

    def step(self):
        # Hyperparameters
        t = self.cur_step / self.num_steps

        # Add noise to dlatents
        noise_strength = self.latent_std * self.initial_noise_factor * \
            max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        latent_noise = (torch.randn_like(self.latent_in)
                        * noise_strength).to(self.device)
        self.latent_expr = self.latent_in + latent_noise

        # Update learning rate
        self.update_lr(t)

        # Train
        self.img_gen, _ = self.g_ema([self.latent_expr], input_is_latent=True, noise=self.noises)
        if self.img_gen.shape[2] > 256:
            self.imag_gen = self.downsample_img(self.img_gen)

        self.loss = self.lpips(self.imag_gen, self.target_images).sum()

        # Noise regularization
        reg_loss = self.noise_regularization()
        self.loss += reg_loss * self.regularize_noise_weight

        # Update params
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

        # Normalize noise
        self.normalize_noise()

    def get_images(self):
        imgs, _ = self.g_ema([self.latent_in], input_is_latent=True,
                             noise=self.noises)
        return imgs
