import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import my_models.model_utils as model_utils

RAIDROOT = os.environ.get('RAIDROOT')


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


class AudioExpressionNet3(nn.Module):
    def __init__(self, T):
        super(AudioExpressionNet3, self).__init__()

        def _set_requires_grad_false(layer):
            for param in layer.parameters():
                param.requires_grad = False

        self.expression_dim = 4 * 512
        self.T = T

        self.convNet = nn.Sequential(
            nn.Conv1d(29, 32, 3, stride=2, padding=1),  # [b, 32, 8]
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),  # [b, 32, 4]
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),  # [b, 64, 2]
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),  # [b, 64, 1]
            nn.LeakyReLU(0.02),
        )

        # Load pre-trained convNet
        self.convNet.load_state_dict(torch.load(
            'model/audio2expression_convNet_justus.pt'))

        latent_dim = 128
        pca_dim = 512
        self.latent_in = nn.Linear(self.expression_dim, latent_dim)
        pca = 'model/audio_dataset_pca512.pt'
        weight = torch.load(pca)[:latent_dim]
        with torch.no_grad():
            self.latent_in.weight = nn.Parameter(weight)

        self.fc1 = nn.Linear(64, 128)
        self.adain1 = model_utils.LinearAdaIN(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, pca_dim)
        self.fc_out = nn.Linear(pca_dim, self.expression_dim)

        # Init fc_out with 512 precomputed pca components
        pca = 'model/audio_dataset_offset_to_mean_4to8_pca512.pt'
        weight = torch.load(pca)[:pca_dim].T
        with torch.no_grad():
            self.fc_out.weight = nn.Parameter(weight)

        # attention
        self.attentionNet = nn.Sequential(
            # b x expression_dim x T => b x 256 x T
            nn.Conv1d(self.expression_dim, 256, 3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 256 x T => b x 64 x T
            nn.Conv1d(256, 64, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 64 x T => b x 16 x T
            nn.Conv1d(64, 16, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 16 x T => b x 4 x T
            nn.Conv1d(16, 4, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            # b x 4 x T => b x 1 x T
            nn.Conv1d(4, 1, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Flatten(),
            nn.Linear(self.T, self.T, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, audio, latent):
        # input shape: [b, T, 16, 29]
        b = audio.shape[0]
        audio = audio.permute(0, 1, 3, 2)  # [b, T, 29, 16]
        audio = audio.view(b * self.T, 29, 16)  # [b * T, 29, 16]

        # Convolution
        conv_res = self.convNet(audio)
        conv_res = conv_res.view(b * self.T, 1, -1)  # [b * T, 1, 64]

        latent = self.latent_in(latent.clone().view(b, -1))

        # Fully connected
        expression = []
        conv_res = conv_res.view(b, self.T, 1, -1)  # [b, T, 1, 64]
        conv_res = conv_res.transpose(0, 1)  # [T, b, 1, 64]
        for t in conv_res:
            z_ = F.leaky_relu(self.adain1(self.fc1(t), latent), 0.02)
            z_ = F.leaky_relu(self.fc2(z_))
            z_ = self.fc3(z_)
            expression.append(self.fc_out(z_))
        expression = torch.stack(expression, dim=1)  # [b, T, expression_dim]

        # expression = expression[:, (self.T // 2):(self.T // 2) + 1]

        if self.T > 1:
            expression_T = expression.transpose(1, 2)  # [b, expression_dim, T]
            attention = self.attentionNet(
                expression_T).unsqueeze(-1)  # [b, T, 1]
            expression = torch.bmm(expression_T, attention)

        return expression.view(b, 4, 512)  # shape: [b, 4, 512]


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
        w = torch.load(RAIDROOT + 'Networks/FERModelGitHub.pt')
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

        self.register_buffer('to_gray', torch.tensor(
            [0.299, 0.587, 0.114]).view(1, 3, 1, 1))

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

    def prepare_img(self, img):
        # Reshape
        b, c, h, w = img.shape
        if w != 48:
            img = nn.functional.interpolate(
                img, 48, mode='bilinear', align_corners=False)
        # Convert to gray
        gray = (img * self.to_gray).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Visualize
        # from torchvision import transforms
        # transforms.ToPILImage('RGB')(img[0].cpu()).show()
        # transforms.ToPILImage('RGB')(gray[0].cpu()).show()
        # 1 / 0

        return gray

    def forward(self, x):
        x = self.prepare_img(x)

        out = self.classifier(x)
        if self.softmaxed:
            out = nn.functional.softmax(out, dim=1)
        out = self._map_to_ravdess_out(out)
        out = self._filter_emotions(out)
        return out


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
        else:
            y = y.view(-1, 1, 512)  # TODO: REMOVE

        return y
