import os
import torch.nn as nn
import torch

from torchvision import models as torch_models

import my_models.model_utils as model_utils


""" Image models """


class VGGStyleClassifier(nn.Module):
    def __init__(self):
        super(VGGStyleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 8),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class FERModelGitHub(nn.Module):
    """
    Source: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
    """
    def __init__(self, pretrained=True):
        super(FERModelGitHub, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, 7)

        if pretrained:
            self._load_weights()

    def _make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
               'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _load_weights(self):
        w = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '../saves/pre-trained/FERModelGitHub.pt'))
        self.load_state_dict(w['net'])

    def _map_to_ravdess_out(self, out):
        ravdess_out = torch.zeros((out.shape[0], 8), dtype=out.dtype, device=out.device)
        ravdess_out[:, 0] = out[:, 6]  # neutral
        # ravdess_out[:, 1] = 0.       # calm
        ravdess_out[:, 2] = out[:, 3]  # happy
        ravdess_out[:, 3] = out[:, 4]  # sad
        ravdess_out[:, 4] = out[:, 0]  # angry
        ravdess_out[:, 5] = out[:, 2]  # fearful
        ravdess_out[:, 6] = out[:, 1]  # disgust
        ravdess_out[:, 7] = out[:, 5]  # surprised
        return ravdess_out

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        out = self._map_to_ravdess_out(out)
        return out


class PreTrainedResNet18(nn.Module):
    def __init__(self, window_size):
        super(PreTrainedResNet18, self).__init__()

        resnet = torch_models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features * window_size

        self.convolutions = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        for i, child in enumerate(self.convolutions.children()):
            for param in child.parameters():
                param.requires_grad = False

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(num_ftrs, 8)

    def forward(self, x):
        y = []
        for idx in range(0, x.shape[1], 3):
            y.append(self.convolutions(x[:, idx:idx + 3]))
        y = torch.cat(y, dim=1)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y


class SiameseConvNet(nn.Module):
    def __init__(self, gray=False):
        super(SiameseConvNet, self).__init__()

        channels = 1 if gray else 3

        self.convolutions = nn.Sequential(
            # shape: [batch_size, 1, 256, 256]
            nn.Conv2d(channels, 16, 5, padding=2),
            # shape: [batch_size, 16, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 16, 128, 128]
            nn.Conv2d(16, 32, 5, padding=2),
            # shape: [batch_size, 32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 32, 64, 64]
            nn.Conv2d(32, 16, 5, padding=2),
            # shape: [batch_size, 16, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 16, 32, 32]
            nn.Conv2d(16, 8, 5, padding=2),
            # shape: [batch_size, 8, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 8, 16, 16]
        )

    def forward(self, x):
        """
        # input shape: [b, sequence_length, 1 | 3, 64, 64]
        # output shape: [b, sequence_length, 16, 8, 8]

        input shape: [b, 1 | 3, 64, 64]
        output shape: [b, 8, 16, 16]
        """
        # y = []
        # for idx in range(x.shape[1]):
        #     y.append(self.convolutions(x[:, idx]))
        # y = torch.stack(y, dim=1)
        # return y
        return self.convolutions(x)


class ConvAndCat(nn.Module):
    def __init__(self, sequence_length, gray):
        super(ConvAndCat, self).__init__()

        self.convolutions = SiameseConvNet(gray)

        # sequence length: 5, params: 527.000
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8 * sequence_length, 8),
        )

    def forward(self, x):
        # shape: [b, sequence_length, c, 64, 64]
        y = self.convolutions(x)
        # shape: [b, sequence_length, 16, 8, 8]
        y = y.view((y.size(0), y.size(1) * y.size(2), y.size(3), y.size(4)))
        # shape: [batch_size, sequence_length * 16, 8, 8]
        y = self.classifier(y)
        # shape: [batch_size, 8]
        return y


class ConvAndPool(nn.Module):
    def __init__(self, gray):
        super(ConvAndPool, self).__init__()

        self.convolutions = SiameseConvNet(gray)

        self.temporal = model_utils.MaxChannelPool()

        # sequence length: 5, params: 527.000
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        # shape: [b, sequence_length, c, h, w]
        y = self.convolutions(x)
        # shape: [b, sequence_length, 16, 8, 8]
        y = self.temporal(y)
        # shape: [b, 16, 8, 8]
        y = self.classifier(y)
        # shape: [b, 8]
        return y


class ConvAnd3D(nn.Module):
    def __init__(self, sequence_length, gray):
        super(ConvAnd3D, self).__init__()

        self.convolutions = SiameseConvNet(gray)

        self.temporal = nn.Conv3d(sequence_length, 1, (5, 5, 5),
                                  padding=(2, 2, 2))

        # sequence length: 5, params: 527.000
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        # shape: [b, sequence_length, c, h, w]
        y = self.convolutions(x)
        # shape: [b, 16, sequence_length, 8, 8]
        y = self.temporal(y)
        # shape: [b, 1, 16, 8, 8]
        y = self.classifier(y)
        # shape: [b, 8]
        return y


class ConvAndRNN(nn.Module):
    def __init__(self, gray):
        super(ConvAndRNN, self).__init__()
        hidden_size = 64
        num_layers = 1

        # Convolutional Layers
        self.convolutions = SiameseConvNet(gray)

        # RNN Layers
        self.temporal = nn.RNN(8 * 8 * 16, hidden_size,
                               num_layers, batch_first=True)

        # sequence length: 5, params: 527.000
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, 8)
        )

    def forward(self, x):
        # shape: [b, sequence_length, c, h, w]
        x = self.convolutions(x)
        # shape: [b, sequence_length, 16, 8, 8]
        x = x.view((*x.shape[:2], -1))
        # shape: [b, sequence_length, 16 * 8 * 8]
        out, _ = self.temporal(x)
        # shape: [b, sequence_length, hidden_size]
        out = self.classifier(out[:, -1])
        # shape: [b, hidden_size]
        return out


class ConvAndConvLSTM(nn.Module):
    def __init__(self, gray):
        super(ConvAndConvLSTM, self).__init__()
        hidden_size = 16

        self.convolutions = SiameseConvNet(gray)

        # 533.600 param version
        self.temporal = model_utils.ConvLSTM(8, hidden_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 16 * 16, 8)
        )

    def forward(self, x):
        # shape: [b, c, h, w]
        y = self.convolutions(x)
        # shape: [b, 8, 16, 16]
        y, _ = self.temporal(y)
        # shape: [b, 8, 8]
        y = self.classifier(y)
        # shape: [b, 8]
        return y


class SiameseConv3D(nn.Module):
    def __init__(self, gray):
        super(SiameseConv3D, self).__init__()

        channels = 1 if gray else 3

        self.convolutions = nn.Sequential(
            # shape: [batch_size, 1, 7, 224, 224]
            nn.Conv3d(channels, 16, (3, 5, 5), padding=(0, 2, 2)),
            # shape: [batch_size, 16, 5, 224, 224]
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            # shape: [batch_size, 16, 5, 112, 112]
            nn.Conv3d(16, 32, (3, 5, 5), padding=(0, 2, 2)),
            # shape: [batch_size, 32, 3, 112, 112]
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            # shape: [batch_size, 32, 3, 56, 56]
            nn.Conv3d(32, 16, (3, 5, 5), padding=(0, 2, 2)),
            # shape: [batch_size, 16, 1, 56, 56]
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            # shape: [batch_size, 16, 1, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 8),
        )

    def forward(self, x):
        # shape: [batch_size, 7, 1, 224, 224]
        y = x.permute(0, 2, 1, 3, 4)
        # shape: [batch_size, 1, 7, 224, 224]
        y = self.convolutions(y)
        # shape: [batch_size, 16, 1, 28, 28]
        y = self.classifier(y)
        # shape: [batch_size, 8]
        return y


class resnetEncoder(nn.Module):
    def __init__(self):
        super(resnetEncoder, self).__init__()

        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.convolutions = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 512 * 18)

    def forward(self, x):

        y = self.convolutions(x)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.linear(y)

        y = y.view(-1, 18, 512)

        return y


class EmotionDatabase(nn.Module):
    def __init__(self, len_dataset, n_latent=16):
        super(EmotionDatabase, self).__init__()
        self.db = nn.Parameter(torch.randn((len_dataset, n_latent)))

    def forward(self, idx):
        return self.db[idx]
