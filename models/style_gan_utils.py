import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

"""
Source: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
"""


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(
                2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=True)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class G_mapping(nn.Sequential):
    def __init__(self, n_latent, length, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        layers = [
            ('pixel_norm', PixelNorm())
        ]
        layers += [('dense', MyLinear(n_latent, n_latent, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
                   ('dense_act', act)] * length
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        return x


class StyleGanBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.noise1 = NoiseLayer(out_channels)
        self.adain1 = StyleMod(latent_size, out_channels)

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.noise2 = NoiseLayer(out_channels)
        self.adain2 = StyleMod(latent_size, out_channels)

    def forward(self, x, latent, noise=None):

        # Upsample
        y = F.interpolate(x, scale_factor=2)

        # Block 1: Conv and norm
        y = self.block1(y)
        # noise
        y = self.noise1(y, noise)
        # AdaIn
        y = self.adain1(y, latent)

        # Block 2: Conv and norm
        y = self.block2(y)
        # noise
        y = self.noise2(y, noise)
        # AdaIn
        y = self.adain2(y, latent)

        return y


class StyleGanInputLayer(nn.Module):
    def __init__(self, channels, latent_size):
        super().__init__()

        self.noise1 = NoiseLayer(channels)
        self.adain1 = StyleMod(latent_size, channels)

        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )
        self.noise2 = NoiseLayer(channels)
        self.adain2 = StyleMod(latent_size, channels)

    def forward(self, x, latent, noise=None):

        # noise
        y = self.noise1(x, noise)
        # AdaIn
        y = self.adain1(y, latent)

        # Block 2: Conv and norm
        y = self.block2(y)
        # noise
        y = self.noise2(y, noise)
        # AdaIn
        y = self.adain2(y, latent)

        return y
