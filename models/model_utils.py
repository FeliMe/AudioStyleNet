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
