import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class MaxChannelPool(nn.Module):
    """
    Max Pool batch of sequences of images along the sequence dimension
    """
    def __init__(self):
        super(MaxChannelPool, self).__init__()

    def forward(self, x):
        b, s, c, w, h = x.size()
        x = x.view(b, s, c * w * h).permute(0, 2, 1)
        y = F.max_pool1d(x, kernel_size=s)
        y = y.permute(0, 2, 1)
        y = y.view(b, c, w, h)
        return y


class Temporal3D(nn.Module):
    def __init__(self, sequence_length, in_channels, out_channels):
        super(Temporal3D, self).__init__()
        assert sequence_length % 2 == 1, "sequence_length must be uneven"
        self.num_conv_layers = sequence_length // 2
        channel_list = [in_channels, *(self.num_conv_layers * [out_channels])]
        self.convs = nn.ModuleList([nn.Conv3d(channel_list[i],
                                              channel_list[i + 1],
                                              (3, 5, 5), padding=(0, 2, 2))
                                    for i in range(self.num_conv_layers)])

    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.convs[i](x)
            x = F.relu(x)
        return x


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM model
    Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size):
        """
        args:
            input_size (int): number of channels of input
            hidden_size (int): number of features in the hidden state h
        """
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3,
                               padding=1)

    def forward(self, x, h_0=None, c_0=None):
        """
        Forward the whole sequence through the ConvLSTM

        args:
            # x (torch.Tensor): input tensor of shape [batch_size,
            #                   sequence_length, channel, height, width]
            # h_0 (torch.Tensor or None): initial hidden state of shape
            #                             [batch_size, hidden_size, channels,
            #                             height, width]
            # c_0 (torch.Tensor or None): initial cell state of shape
            #                             [batch_size, hidden_size, channels,
            #                             height, width]

            x (torch.Tensor): input tensor of shape [batch_size,
                              sequence_length, channel, height, width]
            h_0 (torch.Tensor or None): initial hidden state of shape
                                        [batch_size, hidden_size, channels,
                                        height, width]
            c_0 (torch.Tensor or None): initial cell state of shape
                                        [batch_size, hidden_size, channels,
                                        height, width]
        """

        # get sizes
        batch_size = x.size(0)
        spatial_size = x.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # generate empty h and c, if None is provided
        h, c = h_0, c_0
        if h is None:
            h = torch.zeros(state_size).to(x.device)
        if c is None:
            c = torch.zeros(state_size).to(x.device)

        # Forward every step in the sequence
        h, c = self.forward_one(x, h, c)

        return h, c

    def forward_one(self, x, prev_hidden, prev_cell):

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class UNetDown(nn.Module):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    U-Net encoder block
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    U-Net decoder block
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Up(nn.Module):
    """
    Up-convolutional block
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Up, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpCondCat(nn.Module):
    """
    Up-convolutional block
    """
    def __init__(self, in_channels, out_channels, feat_size, dropout=0.0,
                 cond_dim=16):
        super(UpCondCat, self).__init__()

        self.in_cond = nn.Linear(cond_dim, feat_size)

        layers = [
            nn.ConvTranspose2d(in_channels + 1, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x, cond):
        # Prepare conditioning
        b, c, w, h = x.size()
        cond = self.in_cond(cond)
        cond = cond.view(b, 1, w, h)
        x = torch.cat((x, cond), dim=1)

        # Forward
        return self.main(x)


class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Upconv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv2d(x)
        return x


class SPADE(nn.Module):
    def __init__(self, out_channels, kernel_size, segmap_nc=1):
        super(SPADE, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(out_channels, affine=False)

        nhidden = 128

        padding = kernel_size // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(segmap_nc, nhidden, kernel_size, 1, padding),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_channels, kernel_size, 1, padding)
        self.mlp_beta = nn.Conv2d(nhidden, out_channels, kernel_size, 1, padding)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Resize segmap

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    """
    Source: https://github.com/NVlabs/SPADE/blob/77197af832fac44ba319179096f38467bac91ec9/models/networks/architecture.py
    """
    def __init__(self, in_channels, out_channels, segmap_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (in_channels != out_channels)
        fmiddle = min(in_channels, out_channels)

        # create conv layers
        self.conv_0 = nn.Conv2d(in_channels, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, out_channels, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # apply spectral norm
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(in_channels, 3, segmap_nc)
        self.norm_1 = SPADE(fmiddle, 3, segmap_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, 1, segmap_nc)

    @staticmethod
    def actvn(x):
        return F.leaky_relu(x, 2e-1)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=True):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=False)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv_1)
            self.conv2 = spectral_norm(self.conv_2)

    @staticmethod
    def actvn(x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x):

        y = F.interpolate(x, scale_factor=2)
        y = self.actvn(self.conv_1(y))
        y = self.actvn(self.conv_2(y))

        return y


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral_norm=True):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False)
        self.downSampler = nn.AvgPool2d(2)  # downsampler

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv_1)
            self.conv2 = spectral_norm(self.conv_2)

    @staticmethod
    def actvn(x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x):
        y = self.actvn(self.conv_1(x))
        y = self.actvn(self.conv_2(y))
        y = self.downSampler(y)

        return y


class DiscriminatorCondCat(nn.Module):
    def __init__(self, in_channels, out_channels, feat_size,
                 normalization=True, cond_dim=16):
        super(DiscriminatorCondCat, self).__init__()

        self.in_cond = nn.Linear(cond_dim, feat_size)

        layers = [nn.Conv2d(in_channels + 1, out_channels, 4, 2, 1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x, cond):
        # Prepare conditioning
        b, c, w, h = x.size()
        cond = self.in_cond(cond)
        cond = cond.view(b, 1, w, h)
        x = torch.cat((x, cond), dim=1)

        # Forward
        return self.main(x)


class AdaIN(nn.Module):
    def __init__(self, latent_size, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.lin = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        # Normalize
        x = self.norm(x)

        # Apply transform
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        style = style.view((-1, 2, self.channels, 1, 1))
        x = x * (style[:, 0] + 1.) + style[:, 1]

        return x


class LinearAdaIN(nn.Module):
    def __init__(self, latent_size, target_size):
        super().__init__()
        self.target_size = target_size
        self.norm = nn.InstanceNorm1d(target_size, affine=False)
        self.lin = nn.Linear(latent_size, target_size * 2)

    def forward(self, x, latent):
        # Normalize
        b = x.shape[0]
        x = self.norm(x.view(b, 1, -1)).view(b, -1)

        # Apply transform
        style = self.lin(latent)  # style => [batch_size, target_size*2]
        style = style.view((-1, 2, self.target_size))
        x = x * (style[:, 0] + 1.) + style[:, 1]

        return x


class AdaINResnetGeneratorBlock(nn.Module):
    """
    Source: https://github.com/NVlabs/SPADE/blob/77197af832fac44ba319179096f38467bac91ec9/models/networks/architecture.py
    """

    def __init__(self, in_channels, out_channels, latent_size):
        super().__init__()
        # Attributes
        self.learned_shortcut = (in_channels != out_channels)
        fmiddle = min(in_channels, out_channels)

        # create conv layers
        self.conv_0 = nn.Conv2d(in_channels, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, out_channels,
                                kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

        # define AdaIN layers
        self.adain_0 = AdaIN(latent_size, in_channels)
        self.adain_1 = AdaIN(latent_size, fmiddle)
        if self.learned_shortcut:
            self.adain_s = AdaIN(latent_size, in_channels)

    @staticmethod
    def actvn(x):
        return F.leaky_relu(x, 2e-1)

    def shortcut(self, x, cond):
        if self.learned_shortcut:
            # x_s = self.conv_s(self.adain_s(x, cond))
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def forward(self, x, cond):
        x_s = self.shortcut(x, cond)

        dx = self.conv_0(self.actvn(self.adain_0(x, cond)))
        dx = self.conv_1(self.actvn(self.adain_1(dx, cond)))

        # dx = self.conv_0(self.actvn(x))
        # dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out


class MultiplicativeGaussianNoise1d(nn.Module):
    """
    Multiplies each input channel with random noise with base and gaussian
    noise as exponent.

    args:
        base (float, required): base for the pow operation
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.register_buffer('noise', torch.zeros((1, 1, 1)))

    def forward(self, x):
        if self.training and self.base != 0:
            b, c = x.shape[:2]
            sampled_noise = torch.pow(
                self.base, self.noise.repeat(b, c, 1).normal_())
            x = x * sampled_noise
        return x


def discriminator_block(in_filters, out_filters, normalization=True):
    """
    Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
    Returns downsampling layers of each discriminator block
    """
    layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def weights_init(m):
    ignore_list = ['ConvBlock']
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname not in ignore_list:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
