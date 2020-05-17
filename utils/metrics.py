import numpy as np
import torch
import torch.nn.functional as F

from facenet_pytorch import MTCNN, InceptionResnetV1
from math import exp, log10
from torch.autograd import Variable


class FaceNetDist:
    def __init__(self, image_size=109):
        self.mtcnn = MTCNN(image_size=image_size)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def __call__(self, img1, img2):
        """
        Mean L1 distance between the facenet embeddings of two images
        args:
            img1 (PIL Image)
            img2 (PIL Image)
        """
        # Align images
        img1_cropped = self.mtcnn(img1)
        img2_cropped = self.mtcnn(img2)

        # Visualize
        # from torchvision import transforms
        # transforms.ToPILImage()(img1_cropped).show()
        # transforms.ToPILImage()(img2_cropped).show()
        # 1 / 0

        # Compute embeddings
        img1_embedding = self.resnet(img1_cropped.unsqueeze(0))
        img2_embedding = self.resnet(img2_cropped.unsqueeze(0))

        dist = F.l1_loss(img1_embedding, img2_embedding).item()

        return dist


class FDBM:
    """
    Frequency Domain Image Blur Measure as in
    'Image Sharpness Measure for Blurred Images in Frequency Domain'
    """

    def __init__(self):
        self.name = 'FDBM'

    @staticmethod
    def __call__(img):
        """
        args:
            img (np.array): cv2 grayscale image of shape M x N and dtype np.uint8
        """
        assert img.ndim == 2

        # 1. Compute Fourier Transformation
        f = np.fft.fft2(img)

        # 2. Shift to center
        fc = np.fft.fftshift(f)

        # Visualize
        # magnitude_spectrum = 20 * np.log(np.abs(fc))
        # import matplotlib.pyplot as plt
        # plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

        # 3. Compute absolute value of fc
        fa = np.abs(fc)

        # 4. Calculate maximum value of the frequency component in F
        m = fa.max()

        # 5. Calculate  the total number of pixels in F whose pixel value > thres
        thres = m / 1000.
        th = (f > thres).sum()

        # 6. Calculate Image Quality measure
        (M, N) = img.shape
        fdbm = th / (M * N)

        return fdbm


class PSNR:
    """Peak Signal to Noise Ratio"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(prediction, target):
        mse = F.mse_loss(prediction, target)
        return 10 * log10(1 / mse.item())


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM(torch.nn.Module):
    """
    Structural similarity index from https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
