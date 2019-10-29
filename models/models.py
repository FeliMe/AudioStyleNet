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
    def __init__(self):
        super(SiameseConvNet, self).__init__()

        self.convolutions = nn.Sequential(
            # shape[batch_size, 1, 224, 224]
            nn.Conv2d(1, 16, 5, padding=2),
            # shape[batch_size, 16, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape[batch_size, 16, 112, 112]
            nn.Conv2d(16, 32, 5, padding=2),
            # shape[batch_size, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape[batch_size, 32, 56, 56]
            nn.Conv2d(32, 16, 5, padding=2),
            # shape[batch_size, 16, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # shape[batch_size, 16, 28, 28]
        )

    def forward(self, x):
        """
        input shape: [batch_size, sequence_length, 1, 224, 224]
        output shape: [batch_size, sequence_length, 16, 28, 28]
        """
        y = []
        for idx in range(x.shape[1]):
            y.append(self.convolutions(x[:, idx]))
        y = torch.stack(y, dim=1)
        return y


class ConvAndLSTM(nn.Module):
    def __init__(self):
        super(ConvAndLSTM, self).__init__()
        hidden_size = 8
        num_layers = 1

        # Convolutional Layers
        self.convolutions = SiameseConvNet()

        # RNN Layers
        self.lstm = nn.LSTM(28 * 28 * 16, hidden_size,
                            num_layers, batch_first=True)

    def forward(self, x):
        # shape: [batch_size, sequence_length, channels, height, width]
        x = self.convolutions(x)
        # shape: [batch_size, sequence_length, 16, 28, 28]
        x = x.view((*x.shape[:2], -1))
        # shape: [batch_size, sequence_length, 16 * 28 * 28]
        out, _ = self.lstm(x)
        # shape: [batch_size, sequence_length, hidden_size]
        out = out[:, -1]
        # shape: [batch_size, hidden_size]
        return out


class ConvAndCat(nn.Module):
    def __init__(self, sequence_length):
        super(ConvAndCat, self).__init__()

        self.convolutions = SiameseConvNet()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 28 * 28 * sequence_length, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        # shape: [batch_size, sequence_length, channels, height, width]
        y = self.convolutions(x)
        # shape: [batch_size, sequence_length, 16, 28, 28]
        y = y.view((y.size(0), y.size(1) * y.size(2), y.size(3), y.size(4)))
        # shape: [batch_size, sequence_length * 16, 28, 28]
        y = self.classifier(y)
        # shape: [batch_size, 8]
        return y


class ConvAndPool(nn.Module):
    def __init__(self, sequence_length):
        super(ConvAndPool, self).__init__()

        self.convolutions = SiameseConvNet()

        self.pool = model_utils.MaxChannelPool()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        # shape: [batch_size, sequence_length, channels, height, width]
        y = self.convolutions(x)
        # shape: [batch_size, sequence_length, 16, 28, 28]
        y = self.pool(y)
        # shape: [batch_size, 16, 28, 28]
        y = self.classifier(y)
        # shape: [batch_size, 8]
        return y


class ConvAnd3D(nn.Module):
    def __init__(self, sequence_length):
        super(ConvAnd3D, self).__init__()

        self.convolutions = SiameseConvNet()

        self.conv3d = nn.Conv3d(sequence_length, 1, (4, 4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        # shape: [batch_size, sequence_length, channels, height, width]
        y = self.convolutions(x)
        # shape: [batch_size, sequence_length, 16, 28, 28]
        y = self.conv3d(y)
        # shape: [batch_size, 1, 13, 25, 25]
        y = self.classifier(y)
        # shape: [batch_size, 8]
        return y


""" Landmark models """


class LandmarksLSTM(nn.Module):
    def __init__(self, window_size):
        super(LandmarksLSTM, self).__init__()
        hidden_size = 128
        num_layers = 1

        self.rnn = nn.LSTM(68 * 2 * window_size, hidden_size, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 8)

    def forward(self, x):
        """
        input shape: [batch, sequence_length, 68 * 2 * window_size]
        output shape: [batch, sequence_length, hidden_size]
        """
        out, _ = self.rnn(x)
        # out = torch.flatten(out, 1)
        out = out[:, -1]
        out = self.fc(out)
        return out
