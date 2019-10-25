import torch.nn as nn
import torch

from torchvision import models as torch_models


# Pre-trained ResNet18
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


class SimpleConvNet(nn.Module):
    def __init__(self, window_size):
        super(SimpleConvNet, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(window_size, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        y = self.convolutions(x)
        y = self.classifier(y)

        return y


class SiameseConvNet(nn.Module):
    def __init__(self, window_size):
        super(SiameseConvNet, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28
        )

        self.avgpool = nn.AdaptiveAvgPool2d((28, 28))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28 * window_size, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        y = []
        subshape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        for idx in range(x.shape[1]):
            y.append(self.convolutions(x[:, idx].reshape(subshape)))
        y = torch.cat(y, dim=1)
        y = self.avgpool(y)
        y = self.classifier(y)

        return y


# RNN
class LandmarksLSTM(nn.Module):
    def __init__(self, window_size, sequence_length):
        super(LandmarksLSTM, self).__init__()
        hidden_size = 128
        num_layers = 1

        self.rnn = nn.LSTM(68 * 2 * window_size, hidden_size, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 8)

    def forward(self, x):     # x.shape = [batch, sequence_length, 68 * 2 * window_size]
        out, _ = self.rnn(x)  # out.shape = [batch, sequence_length, hidden_size]
        # out = torch.flatten(out, 1)
        out = out[:, -1]
        out = self.fc(out)
        return out


class ImageLSTM(nn.Module):
    def __init__(self, window_size, sequence_length):
        super(ImageLSTM, self).__init__()
        pass

    def forward(self, x):
        return x
