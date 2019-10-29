import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxChannelPool(nn.Module):
    """
    input shape: [batch_size, sequence_length, channels, height, width]
    output shape: [batch_size,
    """
    def __init__(self):
        super(MaxChannelPool, self).__init__()

    def forward(self, x):
        b, s, c, w, h = x.size()
        x = x.view(b, s, c*w*h).permute(0, 2, 1)
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
        sequence_length = x.size(1)
        spatial_size = x.size()[3:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # generate empty h and c, if None is provided
        h, c = h_0, c_0
        if h is None:
            h = torch.zeros(state_size).to(x.device)
        if c is None:
            c = torch.zeros(state_size).to(x.device)

        # Forward every step in the sequence
        for i in range(sequence_length):
            h, c = self.forward_one(x[:, i], h, c)

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
