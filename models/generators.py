import torch.nn as nn
import torch

import models.model_utils as mu


class GeneratorUNet(nn.Module):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    Pix2Pix U-Net generator
    """
    def __init__(self, gray, num_conditioning_classes):
        super(GeneratorUNet, self).__init__()

        channels = 1 if gray else 3
        latent_channels = 512

        self.num_conditioning_classes = num_conditioning_classes

        # Encoder
        self.down1 = mu.UNetDown(channels, 64, normalize=False)
        self.down2 = mu.UNetDown(64, 128)
        self.down3 = mu.UNetDown(128, 256)
        self.down4 = mu.UNetDown(256, 512, dropout=0.5)
        self.down5 = mu.UNetDown(512, 512, dropout=0.5)
        self.down6 = mu.UNetDown(512, latent_channels, normalize=False, dropout=0.5)

        # Conditioning
        if self.num_conditioning_classes:
            self.embedding = nn.Embedding(num_conditioning_classes, latent_channels)
            latent_channels *= 2

        # Decoder
        self.up1 = mu.UNetUp(latent_channels, 512, dropout=0.5)
        self.up2 = mu.UNetUp(1024, 512, dropout=0.5)
        self.up3 = mu.UNetUp(1024, 256)
        self.up4 = mu.UNetUp(512, 128)
        self.up5 = mu.UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, cond):
        """
        x.shape -> [b, c, h, w]
        cond.shape -> [b, 1]

        args:
            x (torch.tensor): input sequence
            cond (torch.tensor): conditioning label
        """
        # U-Net generator with skip connections from encoder to decoder

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # Conditioning
        if self.num_conditioning_classes:
            emb = self.embedding(cond).view(*d6.size())
            d6 = torch.cat((emb, d6), 1)

        # Decoder
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class SequenceGenerator(nn.Module):
    def __init__(self, gray, num_conditioning_classes):
        super(SequenceGenerator, self).__init__()

        # self.g = GeneratorUNet(gray, num_conditioning_classes)
        self.g = NoiseGenerator(gray)

    def forward(self, x):
        """
        x.shape -> [b, sequence_length, c, h, w]
        cond.shape -> [b, 1]

        args:
            x (torch.tensor): input sequence
            cond(torch.tensor): conditioning label
        """
        y = []
        for idx in range(x.size(1)):
            y.append(self.g(x[:, idx]))
        y = torch.stack(y, dim=1)
        return y


class NoiseGenerator(nn.Module):
    def __init__(self, gray, n_features=64, n_latent=100):
        super(NoiseGenerator, self).__init__()

        nc = 1 if gray else 3
        self.n_latent = n_latent

        # # Conditioning
        # if self.num_conditioning_classes:
        #     self.embedding = nn.Embedding(num_conditioning_classes, n_latent)
        #     n_latent *= 2

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_latent, n_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.ReLU(True),
            # state size. (n_features*8) x 4 x 4
            nn.ConvTranspose2d(n_features * 8, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(True),
            # state size. (n_features*4) x 8 x 8
            nn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(True),
            # state size. (n_features*2) x 16 x 16
            nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(True),
            # state size. (n_features) x 32 x 32
            nn.ConvTranspose2d(n_features, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x, cond):

        # Generate noise
        noise = torch.randn(x.size(0), self.n_latent, 1, 1, device=x.device)

        # # Conditioning
        # if self.num_conditioning_classes:
        #     emb = self.embedding(cond).view(*noise.size())
        #     noise = torch.cat((emb, noise), 1)

        return self.main(noise)
