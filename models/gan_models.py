import torch.nn as nn
import torch

import models.model_utils as mu


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorUNet(nn.Module):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    Pix2Pix U-Net generator
    """
    def __init__(self, gray):
        super(GeneratorUNet, self).__init__()

        channels = 1 if gray else 3

        # Encoder
        self.down1 = mu.UNetDown(channels, 64, normalize=False)
        self.down2 = mu.UNetDown(64, 128)
        self.down3 = mu.UNetDown(128, 256)
        self.down4 = mu.UNetDown(256, 512, dropout=0.5)
        self.down5 = mu.UNetDown(512, 512, dropout=0.5)
        self.down6 = mu.UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder
        self.up1 = mu.UNetUp(512, 512, dropout=0.5)
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

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class SequenceGeneratorUNet(nn.Module):
    def __init__(self, gray):
        super(SequenceGeneratorUNet, self).__init__()

        self.unet = GeneratorUNet(gray)

    def forward(self, x):
        """
        input shape: [batch_size, sequence_length, c, h, w]
        """
        y = []
        for idx in range(x.shape[1]):
            y.append(self.unet(x[:, idx]))
        y = torch.stack(y, dim=1)
        return y


class SequenceDiscriminator(nn.Module):
    def __init__(self, sequence_length, gray):
        super(SequenceDiscriminator, self).__init__()

        self.channels = 1 if gray else 3

        self.model = nn.Sequential(
            *mu.discriminator_block(sequence_length * self.channels * 2, 64,
                                    normalization=False),
            *mu.discriminator_block(64, 128),
            # *mu.discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1, bias=False)

        )

    def forward(self, img_a, img_b):
        """
        a.shape -> [b, sequence_length, c, h, w]
        b.shape -> [b, sequence_length, c, h, w]
        img_input.shape -> [b, 2 * sequence_length * c, h, w]
        """
        img_input = torch.cat((img_a, img_b), 1)
        b, s, c, h, w = img_input.size()
        img_input = img_input.view(b, s * c, h, w)

        patch_size = (b, 1, h // 2 ** 2, w // 2 ** 2)
        return self.model(img_input), patch_size
