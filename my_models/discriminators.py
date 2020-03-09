import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from math import log

import my_models.model_utils as mu
import my_models.style_gan as sg

"""
Shape of real and fake images:
(scales, batch, c, h, w)
"""


class SimpleDiscriminator128(nn.Module):
    def __init__(self):
        super().__init__()

        n_features = 64

        self.d = nn.Sequential(
            # input is 128
            *mu.discriminator_block(3, n_features, normalization=False),
            # state size 64
            *mu.discriminator_block(n_features, n_features, normalization=True),
            # state size 32
            *mu.discriminator_block(n_features, 2 * n_features, normalization=True),
            # state size 16
            *mu.discriminator_block(n_features * 2, n_features * 4, normalization=True),
            # state size 8
            *mu.discriminator_block(n_features * 4, n_features * 8, normalization=True),
            # state size 4
            nn.Conv2d(n_features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.d(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, config):
        super(ConditionalDiscriminator, self).__init__()

        self.n_classes_cond = config.n_classes_cond
        self.pair = config.pair
        n_features = config.n_features_d

        nc = 1 if config.use_gray else 3

        if self.pair:
            nc *= 2

        if self.n_classes_cond:
            self.cond_in = nn.Linear(self.n_classes_cond, 64 * 64 * nc)
            nc *= 2

        self.d = nn.Sequential(
            # input is (nc) x 64 x 64
            *mu.discriminator_block(nc, n_features, normalization=False),
            # state size. (n_features) x 32 x 32
            *mu.discriminator_block(n_features, 2 * n_features, normalization=True),
            # state size. (n_features*2) x 16 x 16
            *mu.discriminator_block(n_features * 2, n_features * 4, normalization=True),
            # state size. (n_features*4) x 8 x 8
            *mu.discriminator_block(n_features * 4, n_features * 8, normalization=True),
            # state size. (n_features*8) x 4 x 4
            nn.Conv2d(n_features * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, inputs):
        img_a = inputs['img_a']
        img_b = inputs['img_b']
        cond = inputs['cond']

        if self.pair:
            x = torch.cat((img_a, img_b), dim=1)
        else:
            x = img_b

        # Conditioning
        if self.n_classes_cond:
            cond = self.cond_in(cond).view(x.size())
            x = torch.cat((x, cond), 1)

        x = self.d(x)

        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, config):
        super(PatchDiscriminator, self).__init__()

        self.n_classes_cond = config.n_classes_cond
        self.pair = config.pair
        n_features = config.n_features.d

        nc = 1 if config.use_gray else 3
        if self.pair:
            nc *= 2

        self.d1 = nn.Sequential(
            # input is (nc) x 64 x 64
            *mu.discriminator_block(nc, n_features, normalization=False),
            # state size. (n_features) x 32 x 32
            *mu.discriminator_block(n_features, n_features * 2),
            # state size. (2 * n_features) x 16 x 16
            *mu.discriminator_block(n_features * 2, n_features * 4),
            # state size. (4 * n_features) x 8 x 8
        )

        # Conditioning
        if self.n_classes_cond:
            self.c_embedding = 4
            self.embedding = nn.Embedding(self.n_classes_cond,
                                          self.c_embedding * 8 * 8)
        else:
            self.c_embedding = 0

        self.d2 = nn.Conv2d(self.c_embedding + n_features * 4, 1, 4, padding=1, bias=False)

    def forward(self, inputs):
        """
        x.shape -> [b, 2 * c, h, w]
        cond.shape -> [b, 1]
        """
        img_a = inputs['img_a']
        img_b = inputs['img_b']
        cond = inputs['cond']

        if self.pair:
            x = torch.cat((img_a, img_b), 1)
        else:
            x = img_b

        x = self.d1(x)

        # Conditioning
        if self.n_classes_cond:
            b, c, h, w = x.size()
            emb = self.embedding(cond).view(b, self.c_embedding, h, w)
            x = torch.cat((x, emb), 1)

        x = self.d2(x)

        return x


class SPADEDiscriminator(nn.Module):
    """
    Source: https://medium.com/@kushajreal/spade-state-of-the-art-in-image-to-image-translation-by-nvidia-bb49f2db2ce3
    """
    def __init__(self, config):
        super().__init__()

        self.pair = config.pair
        n_features = config.n_features_d

        nc = 1 if config.use_gray else 3
        if self.pair:
            nc *= 2

        self.layer1 = self.custom_model1(nc, n_features)
        self.layer2 = self.custom_model2(n_features, 2 * n_features)
        self.layer3 = self.custom_model2(2 * n_features, 4 * n_features)
        self.layer4 = self.custom_model2(4 * n_features, 8 * n_features, stride=1)
        self.inst_norm = nn.InstanceNorm2d(8 * n_features)
        self.conv = spectral_norm(nn.Conv2d(8 * n_features, 1, kernel_size=(4, 4), padding=1))

    @staticmethod
    def custom_model1(in_chan, out_chan):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4, 4), stride=2, padding=1)),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def custom_model2(in_chan, out_chan, stride=2):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4, 4), stride=stride, padding=1)),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, inputs):
        img_a = inputs['img_a']
        img_b = inputs['img_b']

        if self.pair:
            x = torch.cat((img_a, img_b), dim=1)
        else:
            x = img_b
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(self.inst_norm(x))
        x = self.conv(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.depth = 6
        self.n_features = config.n_features_d

        nc = 1 if config.use_gray else 3

        def from_rgb(out_channels):
            return nn.Conv2d(nc, out_channels, (1, 1), bias=False)

        def d_block(in_channels, out_channels):
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),

        self.rgb_to_features = nn.ModuleList([from_rgb(self.n_features // 2)])
        self.layers = nn.ModuleList(
            [mu.DiscriminatorBlock(self.n_features, self.n_features)])

        for i in range(self.depth - 1):
            if i > 2:
                layer = mu.DiscriminatorBlock(
                    int(self.n_features // (2**(i - 2))),
                    int(self.n_features // (2**(i - 2))),
                    use_spectral_norm=True
                )
                rgb = from_rgb(int(self.n_features // (2**(i - 1))))
            else:
                layer = mu.DiscriminatorBlock(self.n_features,
                                              self.n_features // 2,
                                              use_spectral_norm=True)
                rgb = from_rgb(self.n_features // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 1] = \
            from_rgb(self.n_features // (2**(i - 2)))

    def forward(self, inputs):
        inp = inputs['img_b']

        if isinstance(inp, torch.Tensor):
            scale_factors = [(1 / 2**i) for i in range(self.depth)]
            inp = [F.interpolate(inp, scale_factor=s) for s in reversed(scale_factors)]

        y = self.rgb_to_features[self.depth - 1](inp[self.depth - 1])
        y = self.layers[self.depth - 1](y)
        for x, block, converter in \
                zip(reversed(inp[:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        return y


class StyleGANDiscriminator(nn.Module):
    def __init__(self, size=1024, pretrained=True):
        super().__init__()

        template = sg.DiscriminatorOriginal()

        self.progression = template.progression
        self.from_rgb = template.from_rgb
        self.linear = template.linear
        self.step = int(log(size, 2)) - 2

        if pretrained:
            self.load_weights(size)

        self.progression = self.progression[8 - self.step:]
        self.from_rgb = self.from_rgb[8 - self.step:]

        self.n_layer = len(self.progression)

    def load_weights(self, size):
        w = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '../saves/stylegan-%dpx-new.model' % size)
        self.load_state_dict(torch.load(w)['discriminator'])

    def forward(self, y, alpha=-1):
        for i in range(self.step, -1, -1):
            index = self.n_layer - i - 1

            if i == self.step:
                out = self.from_rgb[index](y)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == self.step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(y, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)

        return out
