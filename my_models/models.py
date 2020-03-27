import FrEIA.framework as Ff
import FrEIA.modules as Fm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as torch_models

import my_models.model_utils as model_utils
import my_models.vgg_face as vgg_face


""" Image models """


MAPPING = {
    'neutral': '01',
    'calm': '02',
    'happy': '03',
    'sad': '04',
    'angry': '05',
    'fearful': '06',
    'disgust': '07',
    'surprised': '08'
}


class InvertibleClassifier(nn.Module):
    def __init__(self):
        super(InvertibleClassifier, self).__init__()

        def subnet_conv(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, 64, 5, padding=2), nn.ReLU(),
                                 nn.Conv2d(64, c_out, 5, padding=2), nn.ReLU())

        nodes = [Ff.InputNode(3, 256, 256, name='input')]
        for k in range(3):
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet_conv, 'clamp': 1.2},
                                 name=f'conv{k}'))

            nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        self.conv_inn = Ff.ReversibleGraphNet(nodes)

        self.classifier = nn.Linear(192 * 32 * 32, 8)

    def forward(self, x):
        z = self.conv_inn(x)
        y = self.classifier(z.view(x.size(0), -1))
        return y, z

    def inverse(self, coeff):
        direction = self.linear.weight * coeff
        direction = self.inn(direction, rev=True).detach().cpu().numpy()
        return direction


class AudioExpressionNet(nn.Module):
    def __init__(self):
        super(AudioExpressionNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(29, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
        )

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02),
            # nn.Linear(256, 512 * 18),
            nn.Linear(256, 512 * 4),
            nn.Tanh(),
        )

    def forward(self, x):
        # input shape: [b, 16, 29]
        x = x.transpose(2, 1)
        # shape: [b, 29, 16]
        x = self.convs(x)
        x = self.fcs(x)
        # x = x.view(-1, 18, 512)
        x = x.view(-1, 4, 512)
        return x


class AudioExpressionNet2(nn.Module):
    def __init__(self, T):
        super(AudioExpressionNet2, self).__init__()

        self.T = T

        self.convs = nn.Sequential(
            nn.Conv1d(29, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.02),
        )

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02),
            nn.Linear(256, 512 * 4),
        )

        self.filter = nn.Sequential(
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 512]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 128]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 32]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 8]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 2]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 2, stride=1, padding=0),  # [b, 8, 1]
            nn.LeakyReLU(0.02),
            nn.Flatten(),
            nn.Linear(T, T),
            nn.Sigmoid()
        )

    def forward(self, audio):
        # input shape: [b, T, 16, 29]
        x = audio.permute(1, 0, 3, 2)  # shape: [T, b, 29, 16]

        # Per frame expression estimation
        z = []
        for window in x:
            z.append(self.fcs(self.convs(window)))
        z = torch.stack(z, dim=1)  # shape: [b, 8, 2048]

        # Filtering
        if z.shape[1] > 1:
            f = self.filter(z).unsqueeze(2)  # shape: [b, 8, 1]
            y = torch.mul(z, f)  # shape: [b, 8, 2048]
            y = y.sum(1)  # shape: [b, 2048]
        else:
            y = z[:, 0]

        return y.view(-1, 4, 512)  # shape: [b, 4, 512]


class AudioExpressionNet3(nn.Module):
    def __init__(self, T):
        super(AudioExpressionNet3, self).__init__()

        self.T = T

        self.convs = nn.Sequential(
            # model_utils.MultiplicativeGaussianNoise1d(base=1.4),
            nn.Conv1d(29, 32, 3, stride=2, padding=1),  # [b, 32, 8]
            nn.LeakyReLU(0.02),
            # model_utils.MultiplicativeGaussianNoise1d(base=1.4),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),  # [b, 32, 4]
            nn.LeakyReLU(0.02),
            # model_utils.MultiplicativeGaussianNoise1d(base=1.4),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),  # [b, 64, 2]
            nn.LeakyReLU(0.02),
            # model_utils.MultiplicativeGaussianNoise1d(base=1.4),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),  # [b, 64, 1]
            nn.LeakyReLU(0.02),
        )

        latent_dim = 128
        self.latent_in = nn.Linear(4 * 512, latent_dim)

        self.fc1 = nn.Linear(64, 128)
        self.adain1 = model_utils.LinearAdaIN(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.adain2 = model_utils.LinearAdaIN(latent_dim, 256)
        self.fc_out = nn.Linear(256, 4 * 512)

        self.filter = nn.Sequential(
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 512]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 128]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 32]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 8]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 5, stride=4, padding=2),  # [b, 8, 2]
            nn.LeakyReLU(0.02),
            nn.Conv1d(T, T, 2, stride=1, padding=0),  # [b, 8, 1]
            nn.LeakyReLU(0.02),
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(T, T),
            nn.Sigmoid()
        )

    def forward(self, audio, latent):
        b = audio.shape[0]
        # input shape: [b, T, 16, 29]
        x = audio.permute(1, 0, 3, 2)  # shape: [T, b, 29, 16]

        latent = self.latent_in(latent.clone().view(b, -1))

        # Per frame expression estimation
        z = []
        for window in x:
            z_ = self.convs(window).view(b, -1)
            z_ = F.leaky_relu(self.adain1(self.fc1(z_), latent), 0.02)
            z_ = F.leaky_relu(self.adain2(self.fc2(z_), latent), 0.02)
            z.append(self.fc_out(z_))
        z = torch.stack(z, dim=1)  # shape: [b, 8, 2048]

        # Filtering
        if z.shape[1] > 1:
            f = self.filter(z).unsqueeze(2)  # shape: [b, 8, 1]
            y = torch.mul(z, f)  # shape: [b, 8, 2048]
            y = y.sum(1)  # shape: [b, 2048]
        else:
            y = z[:, 0]

        return y.view(-1, 4, 512)  # shape: [b, 4, 512]


class VGG_Face_VA(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        vgg = vgg_face.VGG_face(pretrained=True)

        self.convs = nn.Sequential(*list(vgg.children())[:-7])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

        if pretrained:
            self._load_weights()

    def _load_weights(self):
        w_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '../saves/pre-trained/vgg_face_va.pt')
        w = torch.load(w_path)
        self.load_state_dict(w)

    def forward(self, x):
        x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
        x = self.convs(x)
        x = self.classifier(x)
        return x


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


class EmotionClassifier(nn.Module):
    def __init__(
        self,
        softmaxed=True,
        emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                  'fearful', 'disgust', 'surprised']):
        super().__init__()
        self.classifier = ConvAndConvLSTM(gray=False)
        self._load_weights()

        self.emotions = [int(MAPPING[e]) - 1 for e in emotions]
        self.softmaxed = softmaxed

        for param in self.classifier.parameters():
            param.requires_grad = False

    def _load_weights(self):
        w = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '../saves/pre-trained/classifier_aligned256.pt'))
        self.classifier.load_state_dict(w)

    def _filter_emotions(self, out):
        return out[:, self.emotions]

    def forward(self, x):
        if x.shape[-1] != 256:
            x = nn.functional.interpolate(
                x, 256, mode='bilinear', align_corners=False)

        out = self.classifier(x)
        if self.softmaxed:
            out = nn.functional.softmax(out, dim=1)
        out = self._filter_emotions(out)
        return out


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

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out


class FERClassifier(nn.Module):
    def __init__(
        self,
        softmaxed=True,
        emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                  'fearful', 'disgust', 'surprised']):
        super().__init__()
        self.classifier = FERModelGitHub(pretrained=True)
        self.emotions = [int(MAPPING[e]) - 1 for e in emotions]
        self.softmaxed = softmaxed

        for param in self.classifier.parameters():
            param.requires_grad = False

    def _map_to_ravdess_out(self, out):
        ravdess_out = torch.zeros(
            (out.shape[0], 8), dtype=out.dtype, device=out.device)
        ravdess_out[:, 0] = out[:, 6]  # neutral
        # ravdess_out[:, 1] = 0.       # calm
        ravdess_out[:, 2] = out[:, 3]  # happy
        ravdess_out[:, 3] = out[:, 4]  # sad
        ravdess_out[:, 4] = out[:, 0]  # angry
        ravdess_out[:, 5] = out[:, 2]  # fearful
        ravdess_out[:, 6] = out[:, 1]  # disgust
        ravdess_out[:, 7] = out[:, 5]  # surprised
        return ravdess_out

    def _filter_emotions(self, out):
        return out[:, self.emotions]

    def forward(self, x):
        if x.shape[-1] != 48:
            x = nn.functional.interpolate(
                x, 48, mode='bilinear', align_corners=False)

        out = self.classifier(x)
        out = self._map_to_ravdess_out(out)
        if self.softmaxed:
            out = nn.functional.softmax(out, dim=1)
        out = self._filter_emotions(out)
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
    def __init__(self, net=18, out_dim=512 * 18, pretrained=False):
        super().__init__()

        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        if net == 18:
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=True)
        elif net == 50:
            from torchvision.models import resnet50
            resnet = resnet50(pretrained=True)

        self.out_dim = out_dim

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        _set_requires_grad_false(self.layer0)
        self.layer1 = resnet.layer1
        _set_requires_grad_false(self.layer1)
        self.layer2 = resnet.layer2
        _set_requires_grad_false(self.layer2)
        self.layer3 = resnet.layer3
        # _set_requires_grad_false(self.layer3)
        self.layer4 = resnet.layer4
        # _set_requires_grad_false(self.layer4)

        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        self.linear_n = nn.Linear(512, out_dim)

        if pretrained:
            self.load_weights()

    def load_weights(self):
        state_dict = torch.load('saves/pre-trained/resNet18Tagesschau.pt')
        self.load_state_dict(state_dict)

    def forward(self, x):

        y = self.layer0(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.linear_n(y)

        if self.out_dim == 18 * 512:
            y = y.view(-1, 18, 512)

        return y


class pretrainedAdaIN(nn.Module):
    def __init__(self, p_norm, latent_size):
        super().__init__()

        # Get original beta and gamma from bn
        mean = p_norm._parameters['bias'].view(1, -1, 1, 1).detach()
        self.register_buffer("mean", mean)
        var = p_norm._parameters['weight'].view(1, -1, 1, 1).detach()
        self.register_buffer("var", var)

        # Own layers
        self.channels = p_norm.num_features
        self.norm = nn.InstanceNorm2d(self.channels)
        self.lin = nn.Linear(latent_size, self.channels * 2)

    def forward(self, x, latent):
        # Normalize
        x = self.norm(x)

        # Apply transform
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        style = style.view((-1, 2, self.channels, 1, 1))
        x = x * (style[:, 0] + self.var) + (self.mean + style[:, 1])

        return x


class pretrainedBasicBlock(nn.Module):
    def __init__(self, p_block, latent_size):
        super().__init__()

        self.conv1 = p_block.conv1
        self.adain1 = pretrainedAdaIN(p_block.bn1, latent_size)
        self.relu = p_block.relu
        self.conv2 = p_block.conv2
        self.adain2 = pretrainedAdaIN(p_block.bn2, latent_size)
        if p_block.downsample is not None:
            self.downsample = True
            self.down_conv = p_block.downsample[0]
            self.down_norm = pretrainedAdaIN(p_block.downsample[1], latent_size)
        else:
            self.downsample = False

    def forward(self, x, score):
        identity = x

        out = self.conv1(x)
        out = self.adain1(out, score)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.adain2(out, score)

        if self.downsample:
            identity = self.down_norm(self.down_conv(x), score)

        out += identity
        out = self.relu(out)

        return out


class pretrainedResNetBlock(nn.Module):
    def __init__(self, p_block, latent_size):
        super().__init__()

        self.block1 = pretrainedBasicBlock(p_block[0], latent_size)
        self.block2 = pretrainedBasicBlock(p_block[1], latent_size)

    def forward(self, x, score):
        x = self.block1(x, score)
        x = self.block2(x, score)

        return x


class neutralToXResNet(nn.Module):
    def __init__(self):
        super().__init__()

        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        # Layer 0
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])  # 64
        _set_requires_grad_false(self.layer0)
        # Layer 1
        # self.layer1 = resnet.layer1  # 64
        self.layer1 = pretrainedResNetBlock(resnet.layer1, 1)
        # Layer 2
        # self.layer2 = resnet.layer2  # 128
        self.layer2 = pretrainedResNetBlock(resnet.layer2, 1)
        # Layer 3
        # self.layer3 = resnet.layer3  # 256
        self.layer3 = pretrainedResNetBlock(resnet.layer3, 1)
        # Layer 4
        # self.layer4 = resnet.layer4  # 512
        self.layer4 = pretrainedResNetBlock(resnet.layer4, 1)

        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        # self.lin_score = nn.Linear(1, 64)
        # self.linear_n = nn.Linear(512 + 64, 512 * 18)
        self.linear_n = nn.Linear(512, 512 * 18)

    def forward(self, x, score):

        y = self.layer0(x)
        y = self.layer1(y, score)
        y = self.layer2(y, score)
        y = self.layer3(y, score)
        y = self.layer4(y, score)
        y = self.avgpool(y)
        y = self.flatten(y)

        # y = torch.cat((y, self.lin_score(score)), dim=1)

        y = self.linear_n(y).view(-1, 18, 512)

        return y


class resNetOffsetEncoder(nn.Module):
    def __init__(self, n_emotions):
        super().__init__()

        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])  # 64
        _set_requires_grad_false(self.layer0)
        self.layer1 = pretrainedResNetBlock(resnet.layer1, n_emotions)
        self.layer2 = pretrainedResNetBlock(resnet.layer2, n_emotions)
        self.layer3 = pretrainedResNetBlock(resnet.layer3, n_emotions)
        self.layer4 = pretrainedResNetBlock(resnet.layer4, n_emotions)

        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        self.linear_n = nn.Linear(512, 512 * 18)
        self.lin_offest = nn.Linear(512, 512 * 18)

    def forward(self, x, score):

        y = self.layer0(x)
        y = self.layer1(y, score)
        y = self.layer2(y, score)
        y = self.layer3(y, score)
        y = self.layer4(y, score)
        y = self.avgpool(y)
        y = self.flatten(y)

        y_n = self.linear_n(y).view(-1, 18, 512)
        y_offset = self.lin_offest(y).view(-1, 18, 512)

        return y_n, y_offset


class neutralToXMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 18 * 512)
        )

    def forward(self, x, score):

        y = self.model(score).view(-1, 18, 512)

        return y


class lmToStyleGANLatent(nn.Module):
    def __init__(self, avg_latent=torch.randn((1, 512))):
        super(lmToStyleGANLatent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3 * 8 * 8 + 68 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 18 * 512)
        )

        self.register_buffer('avg_latent', avg_latent)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        latent_offset = self.model(x).view(-1, 18, 512)
        return latent_offset + self.avg_latent


class resNetAdaINEncoder(nn.Module):
    def __init__(self, n_factors, n_latent, len_dataset):
        super().__init__()

        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)

        self.fc_factor = nn.Linear(n_factors, n_latent)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        _set_requires_grad_false(self.layer0)
        self.layer1 = resnet.layer1
        _set_requires_grad_false(self.layer1)
        self.layer2 = resnet.layer2
        _set_requires_grad_false(self.layer2)
        self.layer3 = pretrainedResNetBlock(resnet.layer3, n_latent)
        self.layer4 = pretrainedResNetBlock(resnet.layer4, n_latent)

        self.avgpool = resnet.avgpool
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 512 * 18)
        # self.linear = nn.Linear(512, 512)

        self.factors = nn.Parameter(torch.rand((len_dataset, n_factors)))

    def forward(self, x, factor, index):

        if factor is None:
            factor = self.factors[index]

        factor = self.fc_factor(factor)

        y = self.layer0(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y, factor)
        y = self.layer4(y, factor)
        y = self.avgpool(y)
        y = self.flatten(y)

        y = self.linear(y).view(-1, 18, 512)
        # y = self.linear(y).view(-1, 1, 512)

        return y


class EmotionDatabase(nn.Module):
    def __init__(self, len_dataset, n_latent=16):
        super(EmotionDatabase, self).__init__()
        self.db = nn.Parameter(torch.randn((len_dataset, n_latent)))

    def forward(self, idx):
        return self.db[idx]
