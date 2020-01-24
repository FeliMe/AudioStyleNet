import argparse
import numpy as np
import os
import torch

from tqdm import tqdm
from my_models.style_gan_2 import Generator
# from lpips import PerceptualLoss, EmotionLoss
from PIL import Image
from utils.perceptual_loss import EmotionClassifier, FERLossLpips, EmotionLoss
from utils.utils import Downsample
from torchvision import transforms
from torchvision.utils import save_image


def update_lr(t, initial_lr, opt):
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
    lr = initial_lr * lr_ramp
    opt.param_groups[0]['lr'] = lr
    return lr


def downsample_img(img):
    b, c, h, w = img.shape
    factor = h // 256
    img = img.reshape(
        b, c, h // factor, factor, w // factor, factor)
    img = img.mean([3, 5])
    return img


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent', type=str, help='Latent to be optimized', required=True)
    parser.add_argument('-t', '--target_image', type=str, help='Target image', required=True)
    parser.add_argument('--dst_dir', type=str, help='Target directory', default='saves/ascent_emotion')
    parser.add_argument('--lr', default=0.1, type=int)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make target directory
    target_dir = args.dst_dir
    os.makedirs(target_dir, exist_ok=True)

    sample = torch.load(args.latent).to(device)
    sample.requires_grad = True

    g = Generator(1024, 512, 8, pretrained=True).to(device)
    g.noises = [n.to(device) for n in g.noises]

    emotion_classifier = EmotionClassifier(use_mask=False).to(device)
    # emotion_classifier = FERModelGitHub(pretrained=True).to(device)
    # criterion = EmotionLoss().to(device)
    # criterion = FERLossLpips().to(device)
    criterion = EmotionLoss().to(device)

    opt = torch.optim.Adam([sample], lr=args.lr)

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

    # Load target image
    # t = transforms.Compose([transforms.Resize(48), transforms.ToTensor()])
    t = transforms.Compose([transforms.ToTensor(), Downsample(256)])
    target = t(Image.open(args.target_image)).unsqueeze(0).to(device)

    # Save target image in target directory
    save_image(target.detach().cpu(), '{}/target_img.png'.format(target_dir),
               normalize=True)

    import torch.nn.functional as F
    pred = F.softmax(emotion_classifier(target), dim=1)[
        0].detach().cpu().numpy()
    print("Target emotion")
    for key, value in emotions.items():
        print("{}: {:.3f}".format(key, pred[value]))
    print()

    # Init pbar
    num_steps = 1000
    i_save = 1
    pbar = tqdm(range(num_steps))

    for i in pbar:
        t = i / num_steps
        lr = update_lr(t, args.lr, opt)

        # Generated image from latent vetor
        img_gen, _ = g([sample], input_is_latent=True, noise=g.noises)

        # Save generated image in interval
        if i % 50 == 0:
            save_image(img_gen.detach().cpu(), '{}/{}.png'.format(
                target_dir, str(i_save).zfill(3)), normalize=True)
            i_save += 1

        # Downsample generated image to 256
        img_gen = downsample_img(img_gen)

        # emotion_pred = emotion_classifier(img_gen)
        # loss = criterion(emotion_pred, target)
        emotion_loss = criterion(img_gen, target)
        loss = emotion_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_description("lr: {:.4f}".format(lr))

        if i % 50 == 0:
            np.set_printoptions(precision=3, suppress=True)
            with torch.no_grad():
                emotion_pred = emotion_classifier(img_gen)[0].cpu().numpy()
            print(f'\nloss: {loss.item(): .3f}')
            for key, value in emotions.items():
                print("{}: {:.3f}".format(key, emotion_pred[value]))

    img_gen, _ = g([sample], input_is_latent=True, noise=g.noises)
    save_image(img_gen.detach().cpu(), '{}/{}.png'.format(
        target_dir, str(i_save).zfill(3)), normalize=True)
