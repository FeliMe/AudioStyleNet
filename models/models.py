import torch.nn as nn
import torch

from torchvision import models as torch_models

import models.model_utils as model_utils


""" Image models """


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
            # shape: [batch_size, 1, 224, 224]
            nn.Conv2d(channels, 16, 5, padding=2),
            # shape: [batch_size, 16, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 16, 112, 112]
            nn.Conv2d(16, 32, 5, padding=2),
            # shape: [batch_size, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 32, 56, 56]
            nn.Conv2d(32, 16, 5, padding=2),
            # shape: [batch_size, 16, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape: [batch_size, 16, 28, 28]
        )

    def forward(self, x):
        """
        input shape: [b, sequence_length, 1 | 3, 64, 64]
        output shape: [b, sequence_length, 16, 8, 8]
        """
        y = []
        for idx in range(x.shape[1]):
            y.append(self.convolutions(x[:, idx]))
        y = torch.stack(y, dim=1)
        return y


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
        hidden_size = 32

        self.convolutions = SiameseConvNet(gray)

        # 533.600 param version
        self.temporal = model_utils.ConvLSTM(16, hidden_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 8 * 8, 8)
        )

    def forward(self, x):
        # shape: [b, sequence_length, c, h, w]
        y = self.convolutions(x)
        # shape: [b, sequence_length, 16, 8, 8]
        y, _ = self.temporal(y)
        # shape: [b, hidden_size, 8, 8]
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


""" Landmark models """


class LandmarksLSTM(nn.Module):
    def __init__(self, window_size):
        super(LandmarksLSTM, self).__init__()
        hidden_size = 256
        num_layers = 3

        self.rnn = nn.LSTM(68 * 2 * window_size, hidden_size, num_layers,
                           batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        # shape: [batch, sequence_length, 68 * 2 * window_size]
        out, _ = self.rnn(x)
        # shape: [batch, sequence_length, hidden_size]
        out = out[:, -1]
        # shape: [batch, hidden_size]
        out = self.classifier(out)
        # shape: [batch, 8]
        return out


class EmotionDatabase(nn.Module):
    def __init__(self, len_dataset, n_latent=16):
        super(EmotionDatabase, self).__init__()
        self.db = nn.Parameter(torch.randn((len_dataset, n_latent)))

    def forward(self, idx):
        return self.db[idx]
