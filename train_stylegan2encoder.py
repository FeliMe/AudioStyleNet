import argparse
import os
import torch

from torchvision.models import resnet18

if __name__ == '__main__':

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = None
