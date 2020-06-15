import torch.nn as nn


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
        x.mul_(style[:, 0] + 1.).add_(style[:, 1])

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
