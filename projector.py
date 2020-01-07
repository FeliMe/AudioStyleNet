import os
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from lpips import PerceptualLoss


class Projector:
    def __init__(self,
                 num_steps=1000,
                 initial_learning_rate=0.1,
                 initial_noise_factor=0.05,
                 ):

        self.num_steps = num_steps
        self.dlatent_avg_samples = 10000
        self.initial_learning_rate = initial_learning_rate
        self.initial_noise_factor = initial_noise_factor
        self.lr_rampdown_length = 0.25
        self.lr_rampup_length = 0.05
        self.noise_ramp_length = 0.75
        self.regularize_noise_weight = 1e5
        self.verbose = True

        self._G = None
        self._device = None
        self._minibatch_size = None
        self._dlatent_avg = None
        self._dlatent_std = None
        self._noise_vars = None
        self._noise_init_op = None
        self._noise_normalize_op = None
        self._dlatents_var = None
        self._noise_in = None
        self._dlatents_expr = None
        self._images_expr = None
        self._target_images_var = None
        self._lpips = None
        self._dist = None
        self._loss = None
        self._reg_sizes = None
        self._lrate_in = None
        self._opt = None
        self._opt_step = None
        self._cur_step = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, g, minibatch_size=1):
        assert minibatch_size == 1
        self._G = g
        self._device = next(g.parameters()).device
        self._minibatch_size = minibatch_size
        if self._G is None:
            return

        # Find dlatent stats
        self._info(('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples))
        torch.manual_seed(123)
        latent_samples = torch.randn((self.dlatent_avg_samples, 512), device=self._device)
        dlatent_samples = self._G.g.style(latent_samples)
        self._dlatent_avg = dlatent_samples.mean(0, keepdim=True)
        self._dlatent_std = (torch.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._dlatents_var = self._dlatent_avg.to(self._device)
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = [v.to(self._device) for v in self._G.noises]
        noise_init_ops = []
        noise_normalize_ops = []
        for v in self._noise_vars:
            noise_init_ops.append(torch.randn(v.shape))
            noise_mean = torch.mean(v)
            noise_std = torch.mean((v - noise_mean) ** 2) ** 0.5
            noise_normalize_ops.append((v - noise_mean) / noise_std)
            # self._info(v)

        # Init optimizer
        self._info('Setting up optimizer...')
        self._opt = torch.optim.Adam(
            [self._dlatents_var.requires_grad_()] + [noise.requires_grad_() for noise in self._noise_vars],
            lr=self.initial_learning_rate)

        # Init loss function
        self._info('Setting up loss function...')
        self._lpips = PerceptualLoss(net='vgg').to(self._device)

    def noise_regularization(self):
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += torch.mean(v * v.roll(1, dims=3)) ** 2 + torch.mean(v * v.roll(1, dims=2)) ** 2
                if sz <= 8:
                    break  # Small enough already
                v = v.view([1, 1, sz // 2, 2, sz // 2, 2])  # Downscale
                v = v.mean(dim=[3, 5])
                sz = sz // 2
        return reg_loss

    def run(self, target_images):
        # Run to completion
        os.makedirs('saves/explore_latent', exist_ok=True)
        self._cur_step = 0
        self._info('Running...')
        self._target_images_var = target_images
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results
        save_image(self.get_images(), 'saves/explore_latent/proj_result.jpg')

    def step(self):
        # Hyperparameters
        t = self._cur_step / self.num_steps

        # Add noise to dlatents
        noise_strength = self._dlatent_std * self.initial_noise_factor * \
            max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        dlatents_noise = (torch.randn_like(self._dlatents_var)
                          * noise_strength).to(self._device)
        self._dlatents_expr = self._dlatents_var + dlatents_noise

        # Update learning rate
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp
        self._opt.param_groups[0]['lr'] = learning_rate

        # Train
        self._opt.zero_grad()
        self._images_expr, _ = self._G.g([self._dlatents_expr], input_is_latent=True, noise=self._noise_vars)
        proc_images_expr = F.interpolate(self._images_expr, size=(
            256, 256), mode='bilinear', align_corners=False)

        self._dist = self._lpips(proc_images_expr, self._target_images_var)
        self._loss = torch.mean(self._dist)

        # Noise regularization
        reg_loss = self.noise_regularization()
        self._loss += reg_loss * self.regularize_noise_weight

        # Update params
        self._loss.backward()
        self._opt.step()

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' %
                       (self._cur_step, self._dist, self._loss))
            # save_image(self.get_images(),
            #            'saves/explore_latent/proj_%d.jpg' % self._cur_step)
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_images(self):
        imgs, _ = self._G.g([self._dlatents_expr],
                            input_is_latent=True, noise=self._noise_vars)
        return imgs
