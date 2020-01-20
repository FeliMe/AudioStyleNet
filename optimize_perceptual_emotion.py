import argparse
import numpy as np
import os
import torch

from tqdm import tqdm
from my_models.style_gan_2 import Generator
from lpips import PerceptualLoss
from PIL import Image
from utils.perceptual_loss import EmotionLoss, EmotionClassifier
from utils.utils import Downsample
from torchvision import transforms
from torchvision.utils import save_image


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent', help='Latent to be optimized', required=True)
    parser.add_argument('-i', '--target_image', help='Target image', required=True)
    parser.add_argument('--dst_dir', help='Target directory', default='saves/ascent_emotion')
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make target directory
    target_dir = args.dst_dir
    os.makedirs(target_dir, exist_ok=True)

    sample = torch.load(args.latent).to(device)
    sample.requires_grad = True

    # mask_predictor = FaceMaskPredictor()
    # mask = None

    g = Generator(1024, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    emotion_classifier = EmotionClassifier(use_mask=False).to(device)
    # criterion = nn.BCELoss()
    criterion = EmotionLoss(use_mask=False).to(device)
    opt = torch.optim.Adam([sample])

    # Regularization
    lpips = PerceptualLoss(model='net-lin', net='vgg').to(device)

    emotions = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgust': 6,
        'surprise': 7
    }

    # target = torch.zeros((1, 8)).to(device)
    # target[:, emotions['angry']] = 1.

    transform = transforms.Compose([
        transforms.ToTensor(),
        Downsample(256),
    ])
    target = transform(Image.open(args.target_image)).unsqueeze(0).to(device)

    save_image(target.cpu(), '{}/target_img.png'.format(target_dir),
               normalize=True)

    # Init pbar
    num_steps = 1000
    pbar = tqdm(range(num_steps))

    for i in pbar:
        # Generated image from latent vetor
        img_gen, _ = g([sample], input_is_latent=True, noise=g.noises)

        # Downsample generated image to 256
        b, c, h, w = img_gen.shape
        factor = h // 256
        img_gen = img_gen.reshape(
            b, c, h // factor, factor, w // factor, factor)
        img_gen = img_gen.mean([3, 5])

        # if mask is None:
        #     mask = mask_predictor.get_mask(img_gen).to(device)

        # emotion_pred = emotion_classifier(img_gen)
        # loss = criterion(emotion_pred, target)
        emotion_loss = criterion(img_gen, target)
        # perceptual_loss = lpips(img_gen, target)
        # loss = emotion_loss + perceptual_loss
        loss = emotion_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 50 == 0:
            np.set_printoptions(precision=3, suppress=True)
            with torch.no_grad():
                emotion_pred = emotion_classifier(img_gen)[0].cpu().numpy()
            print(f'\nloss: {loss.item(): .3f}')
            for key, value in emotions.items():
                print("{}: {:.3f}".format(key, emotion_pred[value]))
            # pbar.set_description("Emotion: {}".format(emotion_pred.cpu().numpy()))
            # pbar.set_description(
            #     (f'e-loss: {emotion_loss.item():.3f}; lpips: {perceptual_loss.item():.3f}'))
            save_image(img_gen.detach().cpu(), '{}/{}_of_{}.png'.format(
                target_dir, i, num_steps), normalize=True)
