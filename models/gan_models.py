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


class SequenceGeneratorUNet(nn.Module):
    def __init__(self, gray, num_conditioning_classes):
        super(SequenceGeneratorUNet, self).__init__()

        self.unet = GeneratorUNet(gray, num_conditioning_classes)

    def forward(self, x, cond):
        """
        x.shape -> [b, sequence_length, c, h, w]
        cond.shape -> [b, 1]

        args:
            x (torch.tensor): input sequence
            cond(torch.tensor): conditioning label
        """
        y = []
        for idx in range(x.size(1)):
            y.append(self.unet(x[:, idx], cond))
        y = torch.stack(y, dim=1)
        return y


class PatchDiscriminator(nn.Module):
    def __init__(self, gray, num_conditioning_classes):
        super(PatchDiscriminator, self).__init__()

        channels = 1 if gray else 3
        self.num_conditioning_classes = num_conditioning_classes

        self.d1 = nn.Sequential(
            *mu.discriminator_block(2 * channels, 8, normalization=False),
            *mu.discriminator_block(8, 8),
            *mu.discriminator_block(8, 16),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )

        # Conditioning
        if self.num_conditioning_classes:
            self.c_embedding = 4
            self.embedding = nn.Embedding(self.num_conditioning_classes,
                                          self.c_embedding * 9 * 9)
        else:
            self.c_embedding = 0

        self.d2 = nn.Conv2d(self.c_embedding + 16, 1, 4, padding=1, bias=False)

    def forward(self, x, cond):
        """
        x.shape -> [b, 2 * c, h, w]
        cond.shape -> [b, 1]
        """
        y = self.d1(x)

        # Conditioning
        if self.num_conditioning_classes:
            b, c, h, w = y.size()
            emb = self.embedding(cond).view(b, self.c_embedding, h, w)
            y = torch.cat((y, emb), 1)

        y = self.d2(y)

        return y


class SequencePatchDiscriminator(nn.Module):
    def __init__(self, gray, num_conditioning_classes):
        super(SequencePatchDiscriminator, self).__init__()

        self.d = PatchDiscriminator(gray, num_conditioning_classes)

    def forward(self, img_a, img_b, cond):
        """
        a.shape -> [b, sequence_length, c, h, w]
        b.shape -> [b, sequence_length, c, h, w]
        cond.shape -> [b, 1]

        img_input.shape -> [b, sequence_length, 2 * c, h, w]

        args:
            img_a (torch.tensor): input image
            img_b (torch.tensor): target image
            cond (torch.tensor): conditioning label
        """
        img_input = torch.cat((img_a, img_b), 2)

        out = []
        for i_seq in range(img_b.size(1)):
            out.append(self.d(img_input[:, i_seq], cond))
        out = torch.stack(out, 1)

        return out


class SequenceDiscriminator(nn.Module):
    def __init__(self, gray):
        super(SequenceDiscriminator, self).__init__()

        channels = 1 if gray else 3

        self.model = nn.Sequential(
            *mu.discriminator_block(channels * 2, 8, normalization=False),
            *mu.discriminator_block(8, 8),
            *mu.discriminator_block(8, 16),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(16, 16, 4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 2),
            nn.Sigmoid()
        )

    def forward(self, img_a, img_b):
        """
        a.shape -> [b, sequence_length, c, h, w]
        b.shape -> [b, sequence_length, c, h, w]
        img_input.shape -> [b, 2 * sequence_length * c, h, w]
        out.shape -> [b, sequence_length, 1]
        """
        img_input = torch.cat((img_a, img_b), 2)

        out = []
        for i_seq in range(img_a.size(1)):
            out.append(self.model(img_input[:, i_seq]))
        out = torch.stack(out, 1)

        return out
