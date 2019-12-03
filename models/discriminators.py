import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

import models.model_utils as mu


class SequenceDiscriminator(nn.Module):
    def __init__(self, d):
        super(SequenceDiscriminator, self).__init__()

        self.d = d

    def forward(self, inputs):
        """
        a.shape -> [b, sequence_length, c, h, w]
        b.shape -> [b, sequence_length, c, h, w]
        img_input.shape -> [b, 2 * sequence_length * c, h, w]
        out.shape -> [b, sequence_length, 1]
        """
        img_a = inputs['img_a']
        img_b = inputs['img_b']
        cond = inputs['cond']

        out = []
        for i_seq in range(img_b.size(1)):
            out.append(self.d({
                'img_a': img_a[:, i_seq],
                'img_b': img_b[:, i_seq],
                'cond': cond
            }))
        out = torch.stack(out, 1)

        return out


class SimpleDiscriminator(nn.Module):
    def __init__(self, gray, n_classes_cond, pair, n_features=64):
        super(SimpleDiscriminator, self).__init__()

        self.n_classes_cond = n_classes_cond
        self.pair = pair

        nc = 1 if gray else 3
        if self.pair:
            nc *= 2

        if self.n_classes_cond:
            self.cond_in = nn.Linear(self.n_classes_cond, 64 * 64 * nc)
            nc *= 2

        self.d = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, n_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features) x 32 x 32
            nn.Conv2d(n_features, n_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*2) x 16 x 16
            nn.Conv2d(n_features * 2, n_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_features*4) x 8 x 8
            nn.Conv2d(n_features * 4, n_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
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
    def __init__(self, gray, n_classes_cond, pair, n_features=8):
        super(PatchDiscriminator, self).__init__()

        self.n_classes_cond = n_classes_cond
        self.pair = pair

        nc = 1 if gray else 3
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
    def __init__(self, gray, n_classes_cond, pair, n_features=64):
        super().__init__()

        self.pair = pair

        nc = 1 if gray else 3
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
        cond = inputs['cond']

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
