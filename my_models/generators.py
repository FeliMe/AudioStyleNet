import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log

import my_models.model_utils as mu
import my_models.style_gan as sg


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
        x = inputs['img_a']

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
        x = inputs['img_a']
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
        segmap = inputs['img_a']

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


class AdaINGenerator128(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        n_features = 64

        self.conv1 = nn.Conv2d(1, 8 * n_features, 3, padding=1)
        # state size 4 x 4
        self.up_0 = mu.AdaINResnetGeneratorBlock(8 * n_features, 4 * n_features, latent_size)
        # state size 8 x 8
        self.up_1 = mu.AdaINResnetGeneratorBlock(4 * n_features, 4 * n_features, latent_size)
        # state size 16 x 16
        self.up_2 = mu.AdaINResnetGeneratorBlock(4 * n_features, 2 * n_features, latent_size)
        # state size 32 x 32
        self.up_3 = mu.AdaINResnetGeneratorBlock(2 * n_features, 2 * n_features, latent_size)
        # state size 64 x 64
        self.up_4 = mu.AdaINResnetGeneratorBlock(2 * n_features, n_features, latent_size)
        # state size 128 x 128
        self.up_5 = mu.AdaINResnetGeneratorBlock(n_features, n_features, latent_size)
        # state size 128 x 128
        self.conv_img = nn.Conv2d(n_features, 3, kernel_size=(3, 3), padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, cond):
        x = torch.randn((cond.shape[0], 1, 4, 4), device=cond.device)

        # downsample segmap and run convolution
        x = self.conv1(x)

        # Generator
        x = self.up_0(x, cond)
        x = self.up(x)
        x = self.up_1(x, cond)
        x = self.up(x)
        x = self.up_2(x, cond)
        x = self.up(x)
        x = self.up_3(x, cond)
        x = self.up(x)
        x = self.up_4(x, cond)
        x = self.up(x)
        x = self.up_5(x, cond)

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
        x = inputs['img_a']

        y = torch.randn(x.size(0), self.n_features, 1, 1, device=x.device)

        outputs = []
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(torch.tanh(converter(y)))

        return outputs


class ProGANDecoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ProGANDecoder, self).__init__()

        main = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                              'PGAN', model_name='celebAHQ-512',
                              pretrained=pretrained).netG

        self.normalizationLayer = main._modules['module'].normalizationLayer
        self.leakyRelu = main._modules['module'].leakyRelu
        self.groupScale0 = main._modules['module'].groupScale0
        self.alpha = main._modules['module'].alpha
        self.scaleLayers = main._modules['module'].scaleLayers
        self.toRGBLayers = main._modules['module'].toRGBLayers
        self.generationActivation = main._modules['module'].generationActivation
        self.formatLayer = main._modules['module'].formatLayer

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # Normalize the input ?
        if self.normalizationLayer is not None:
            x = self.normalizationLayer(x)
        x = x.view(-1, self.num_flat_features(x))
        # format layer
        x = self.leakyRelu(self.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.normalizationLayer(x)

        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            x = self.leakyRelu(convLayer(x))
            if self.normalizationLayer is not None:
                x = self.normalizationLayer(x)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](x)
            # y = sgu.upscale2d(y)
            y = F.interpolate(y, scale_factor=2, mode='nearest')

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):

            # x = sgu.upscale2d(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                # y = sgu.upscale2d(y)
                y = F.interpolate(y, scale_factor=2, mode='nearest')

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        print(x.shape)

        return x


class StyleGANDecoder(nn.Module):
    def __init__(self, size=1024, pretrained=True):
        super(StyleGANDecoder, self).__init__()

        template = sg.StyledGeneratorOriginal(512)

        self.style = template.style
        self.generator = template.generator
        self.step = int(log(size, 2)) - 2

        if pretrained:
            self.load_weights(size)

        # del self.style

        self.generator.progression = self.generator.progression[:self.step + 1]
        self.generator.to_rgb = self.generator.to_rgb[:self.step + 1]

    def load_weights(self, size):
        w = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '../saves/pre-trained/stylegan-%dpx-new.model' % size)
        self.load_state_dict(torch.load(w)['g_running'])

    def mean_style(self, z):
        return self.style(z).mean(0, keepdim=True)

    def std_style(self, z):
        return self.style(z).std()

    def forward(self, w):
        if type(w) not in (tuple, list):
            w = [w]

        noise = []
        for i in range(self.step + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(w[0].shape[0], 1, size,
                                     size, device=w[0].device))

        y = self.generator(w, noise, self.step)
        return y

    @torch.no_grad()
    def get_mean_std_w(self, device):
        mean_w = None
        for i in range(10):
            w = self.mean_style(torch.randn(1024, 512).to(device))
            if mean_w is None:
                mean_w = w
            else:
                mean_w += w
        std_w = self.std_style(torch.randn(1024, 512).to(device))

        mean_w /= 10
        return mean_w, std_w
