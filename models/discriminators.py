import torch
import torch.nn as nn

import models.model_utils as mu


class PatchDiscriminator(nn.Module):
    def __init__(self, gray, num_conditioning_classes):
        super(PatchDiscriminator, self).__init__()

        channels = 1 if gray else 3
        self.num_conditioning_classes = num_conditioning_classes

        # self.d1 = nn.Sequential(
        #     *mu.discriminator_block(2 * channels, 8, normalization=False),
        #     *mu.discriminator_block(8, 8),
        #     *mu.discriminator_block(8, 16),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        # )

        self.d1 = nn.Sequential(
            *mu.discriminator_block(2 * channels, 32, normalization=False),
            *mu.discriminator_block(32, 32),
            *mu.discriminator_block(32, 64),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )

        # Conditioning
        if self.num_conditioning_classes:
            self.c_embedding = 4
            self.embedding = nn.Embedding(self.num_conditioning_classes,
                                          self.c_embedding * 9 * 9)
        else:
            self.c_embedding = 0

        self.d2 = nn.Conv2d(self.c_embedding + 64, 1, 4, padding=1, bias=False)

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


class SimpleDiscriminator(nn.Module):
    def __init__(self, gray, n_features=64):
        super(SimpleDiscriminator, self).__init__()

        nc = 1 if gray else 3

        self.model = nn.Sequential(
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
            nn.Sigmoid()
        )

    def forward(self, img_a, img_b):
        out = []
        for i_seq in range(img_a.size(1)):
            out.append(self.model(img_b[:, i_seq]))
        out = torch.stack(out, 1)

        return out
