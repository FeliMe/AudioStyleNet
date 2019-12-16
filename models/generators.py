import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_utils as mu
import models.style_gan_utils as sgu


class NoiseGenerator(nn.Module):
    """
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    Generator which generates image from random noise
    """
    def __init__(self, config):
        super(NoiseGenerator, self).__init__()

        gray = config.use_gray
        n_features = config.n_features_g
        n_latent = config.n_latent_noise

        nc = 1 if gray else 3
        self.n_latent = in_channels = n_latent

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, n_features * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(n_features * 8),
            nn.ReLU(True),
            # state size. (n_features*8) x 4 x 4
            mu.Up(n_features * 8, n_features * 4),
            # state size. (n_features*4) x 8 x 8
            mu.Up(n_features * 4, n_features * 2),
            # state size. (n_features*2) x 16 x 16
            mu.Up(n_features * 2, n_features),
            # state size. (n_features) x 32 x 32
            nn.ConvTranspose2d(n_features, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        # Unpack inputs
        x = inputs['x']

        # Generate noise
        noise = torch.randn(x.size(0), self.n_latent, 1, 1, device=x.device)

        return self.main(noise)


class UNetGenerator(nn.Module):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    Pix2Pix U-Net generator
    """
    def __init__(self, config):
        super(UNetGenerator, self).__init__()

        gray = config.use_gray
        n_features = config.n_features_g
        self.n_classes_cond = config.n_classes_cond

        nc = 1 if gray else 3
        latent_channels = n_features * 2

        # Encoder
        self.down1 = mu.UNetDown(nc, n_features, normalize=False)
        self.down2 = mu.UNetDown(n_features, n_features * 2)
        self.down3 = mu.UNetDown(n_features * 2, n_features * 4)
        self.down4 = mu.UNetDown(n_features * 4, n_features * 4, dropout=0.5)
        self.down5 = mu.UNetDown(n_features * 4, n_features * 4, dropout=0.5)
        self.down6 = mu.UNetDown(n_features * 4, latent_channels, normalize=False, dropout=0.5)

        # Conditioning
        if self.num_conditioning_classes:
            self.embedding = nn.Embedding(self.n_classes_cond, latent_channels)
            latent_channels *= 2

        # Decoder
        self.up1 = mu.UNetUp(latent_channels, n_features * 4, dropout=0.5)
        self.up2 = mu.UNetUp(n_features * 8, n_features * 4, dropout=0.5)
        self.up3 = mu.UNetUp(n_features * 8, n_features * 4)
        self.up4 = mu.UNetUp(n_features * 8, n_features * 2)
        self.up5 = mu.UNetUp(n_features * 4, n_features)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, nc, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        """
        x.shape -> [b, c, h, w]
        cond.shape -> [b, 1]

        args:
            x (torch.tensor): input sequence
            cond (torch.tensor): conditioning label
        """
        # Unpack inputs
        x = inputs['x']
        cond = inputs['cond']

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # Conditioning
        if self.self.n_classes_cond:
            emb = self.embedding(cond).view(*d6.size())
            d6 = torch.cat((emb, d6), 1)

        # Decoder
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class SPADEGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        gray = config.use_gray
        n_features = config.n_features_g
        self.n_classes_cond = config.n_classes_cond

        self.h = self.w = 64 // (2 ** 5)

        in_out_ch = 1 if gray else 3

        self.fc = nn.Conv2d(in_out_ch, 8 * n_features, 3, padding=1)
        # state size. (n_features*8) x 2 x 2
        self.up_0 = mu.SPADEResnetBlock(8 * n_features, 8 * n_features, in_out_ch)
        # state size. (n_features*8) x 4 x 4
        self.up_1 = mu.SPADEResnetBlock(8 * n_features, 8 * n_features, in_out_ch)
        # state size. (n_features*8) x 8 x 8
        self.up_2 = mu.SPADEResnetBlock(8 * n_features, 8 * n_features, in_out_ch)
        # state size. (n_features*8) x 16 x 16
        self.up_3 = mu.SPADEResnetBlock(8 * n_features, 4 * n_features, in_out_ch)
        # state size. (n_features*4) x 32 x 32
        self.up_4 = mu.SPADEResnetBlock(4 * n_features, 2 * n_features, in_out_ch)
        # state size. (n_features*2) x 64 x 64
        self.up_5 = mu.SPADEResnetBlock(2 * n_features, n_features, in_out_ch)
        # state size. (n_features) x 64 x 64
        self.conv_img = nn.Conv2d(n_features, in_out_ch, kernel_size=(3, 3), padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, inputs):
        # Unpack inputs
        segmap = inputs['x']

        # downsample segmap and run convolution
        x = F.interpolate(segmap, size=(self.h, self.w))
        x = self.fc(x)

        # Generator
        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)
        x = self.up(x)
        x = self.up_4(x, segmap)
        x = self.up(x)
        x = self.up_5(x, segmap)

        # Create final output
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class MultiScaleGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        gray = config.use_gray
        self.n_features = config.n_features_g
        self.depth = 6

        nc = 1 if gray else 3

        # To RGB for every sublayer
        def to_rgb(in_channels):
            return nn.Conv2d(in_channels, nc, (1, 1), bias=False)

        self.layers = nn.ModuleList(
            [mu.GeneratorBlock(self.n_features, self.n_features)])
        self.rgb_converters = nn.ModuleList([to_rgb(self.n_features)])

        for i in range(self.depth - 1):
            if i <= 2:
                layer = mu.GeneratorBlock(self.n_features, self.n_features,
                                          use_spectral_norm=True)
                rgb = to_rgb(self.n_features)
            else:
                layer = mu.GeneratorBlock(
                    int(self.n_features // (2**(i - 3))),
                    int(self.n_features // (2**(i - 2))),
                    use_spectral_norm=True
                )
                rgb = to_rgb(int(self.n_features // (2**(i - 2))))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, inputs):
        # Unpack inputs
        x = inputs['x']

        y = torch.randn(x.size(0), self.n_features, 1, 1, device=x.device)

        outputs = []
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(torch.tanh(converter(y)))

        return outputs


class StyeGanGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        gray = config.use_gray
        n_features = config.n_features_g
        n_latent = config.n_latent_noise

        nc = 1 if gray else 3
        self.n_latent = n_latent

        self.transform_latent = sgu.G_mapping(n_latent, 4)

        self.const_in = nn.Parameter(torch.ones(1, n_features, 4, 4))
        self.inp_layer = sgu.StyleGanInputLayer(n_features, n_latent)

        # size: (n_features, 4, 4)
        self.block1 = sgu.StyleGanBlock(n_features, n_features, n_latent)
        # size: (n_features, 8, 8)
        self.block2 = sgu.StyleGanBlock(n_features, n_features, n_latent)
        # size: (n_features, 16, 16)
        self.block3 = sgu.StyleGanBlock(n_features, n_features // 2, n_latent)
        # size: (n_features, 32, 32)
        self.block4 = sgu.StyleGanBlock(n_features // 2, n_features // 4, n_latent)
        # size: (n_features, 64, 64)
        self.to_rgb = nn.Conv2d(n_features // 4, nc, 3, padding=1)

    def forward(self, inputs):
        # Unpack inputs
        x = inputs['x']

        # Transform latent noise vector
        latent = torch.randn(x.size(0), self.n_latent, device=x.device)
        latent = self.transform_latent(latent)

        # Synthesize image
        y = self.inp_layer(self.const_in, latent)
        y = self.block1(y, latent)
        y = self.block2(y, latent)
        y = self.block3(y, latent)
        y = self.block4(y, latent)
        y = self.to_rgb(y)
        y = torch.tanh(y)

        return y


# class SequenceGenerator(nn.Module):
#     def __init__(self, g):
#         super(SequenceGenerator, self).__init__()

#         self.g = g

#     def forward(self, x, cond):
#         """
#         x.shape -> [b, sequence_length, c, h, w]
#         cond.shape -> [b, 1]

#         args:
#             x (torch.tensor): input sequence
#             cond(torch.tensor): conditioning label
#         """
#         y = []
#         for idx in range(x.size(1)):
#             y.append(self.g(x[:, idx], cond))

#         if not type(y[0]) is list:
#             y = torch.stack(y, dim=1)
#         return y
