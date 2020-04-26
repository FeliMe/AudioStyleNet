import torch.nn.functional as F

from math import log10


class PSNR:
    """Peak Signal to Noise Ratio"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(prediction, target):
        mse = F.mse_loss(prediction, target)
        return 10 * log10(1 / mse.item())
