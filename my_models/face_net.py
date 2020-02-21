import torch
import torch.nn as nn

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
from utils.utils import Downsample


class FaceNetDistance(nn.Module):
    def __init__(self, device, image_size=256):
        super().__init__()
        self.device = device
        self.mtcnn = MTCNN(image_size=image_size, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(
            pretrained='vggface2').eval().to(self.device)

    def distance(self, img1, img2, percentage=False, embedding=False):
        img1 = self.mtcnn(transforms.ToPILImage(mode='RGB')(img1))
        img2 = self.mtcnn(transforms.ToPILImage(mode='RGB')(img2))

        if img1 is None or img2 is None:
            dist = 2
            embedding_list = [torch.zeros(512), torch.zeros(512)]
        else:
            img_stack = torch.stack([img1, img2]).to(self.device)
            embedding_list = self.resnet(img_stack)
            embedding_dist = embedding_list[0] - embedding_list[1]
            dist = embedding_dist.norm().item()

        if percentage:
            return dist, ((2 - dist) * 50)
        elif embedding:
            return dist, embedding_list
        else:
            return dist


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = transforms.Compose([
        transforms.ToTensor(),
        Downsample(256)
    ])

    fn_dist = FaceNetDistance(device, 256)

    img1 = t(Image.open('saves/test_images/happy.png'))
    img2 = t(Image.open('saves/test_images/angry.png'))
    img3 = t(Image.open('saves/test_images/felix.png'))

    print(fn_dist.distance(img1, img2))
    print(fn_dist.distance(img1, img3))
    print(fn_dist.distance(img2, img3))
