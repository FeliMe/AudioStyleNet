import argparse
import numpy as np
import torch

from my_models import models
from my_models.style_gan_2 import Generator
from PIL import ImageDraw
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm


MAPPING = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}


def int_to_one_hot(labels):
    one_hots = []
    for label in labels:
        one_hot = torch.zeros(8)
        one_hot[label] = 1
        one_hots.append(one_hot)
    return torch.stack(one_hots, dim=0)


def downsample_256(img):
    b, c, h, w = img.shape
    factor = h // 256
    img = img.reshape(b, c, h // factor, factor, w // factor, factor)
    img = img.mean([3, 5])
    return img


if __name__ == '__main__':

    # Parse agruments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-t', '--truncation', type=float, default=0.8)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)

    # Init Classifier
    # classifier = models.EmotionClassifier().eval().to(device)
    classifier = models.FERClassifier().to(device)

    # Generate images
    imgs = []
    scores = []
    labels = []
    labels_one_hot = []
    for i in tqdm(range(args.num_samples // 10)):
        with torch.no_grad():
            inp = torch.randn((10, 512), device=device)
            img, _ = g([inp], truncation=args.truncation,
                       truncation_latent=g.latent_avg.to(device))
            img = downsample_256(img)
            score = classifier(img).cpu()
            label = score.max(dim=1)[1]
            label_one_hot = int_to_one_hot(label)
            img = img.cpu()
        imgs.append(img)
        scores.append(score)
        labels.append(label)
        labels_one_hot.append(label_one_hot)

    # Delete generator to free GPU memory
    del g
    del classifier

    imgs = torch.cat(imgs, dim=0)
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)
    labels_one_hot = torch.cat(labels_one_hot, dim=0)
    print(imgs.shape)

    # Get distribution
    labels_one_hot = labels_one_hot.sum(0)

    # Max and mean scores for each label
    for i in range(8):
        print("{} - # {}; % {:.2f}; max: {:.4f}; mean: {:.4f}; > 0.8: {}".format(
            MAPPING[i].ljust(9, ' '),
            str(int(labels_one_hot[i].item())).ljust(5, ' '),
            labels_one_hot[i] / labels_one_hot.sum(),
            scores[:, i].max(),
            scores[:, i].mean(),
            torch.where(scores[:, i] > 0.8, torch.ones(
                1,), torch.zeros(1,)).sum().item()
        ))

    # Display images with max scores
    for i in range(8):
        np.set_printoptions(precision=3)
        max_idx = scores[:, i].max(dim=0)[1]
        pil_img = transforms.ToPILImage('RGB')(
            make_grid(imgs[max_idx], normalize=True, range=(-1, 1)))
        draw = ImageDraw.Draw(pil_img)
        title = "Max score {}, Label: {}\nscore: {:.4f}\nall scores: {}".format(
            MAPPING[i],
            MAPPING[labels[max_idx].item()],
            scores[max_idx, i],
            scores[max_idx].numpy()
        )
        draw.text((0, 0), title, (255, 255, 255))
        pil_img.show()
