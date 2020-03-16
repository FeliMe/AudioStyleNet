import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
from my_models import models
from my_models import style_gan_2
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from utils import utils


MAPPING = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}
TO_PIL = transforms.ToPILImage('RGB')
TO_TENSOR = transforms.ToTensor()


class LMClassificationModel(nn.Module):
    def __init__(self, dim):
        super(LMClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class LMClassificationSystem(pl.LightningModule):
    def __init__(self, X, y, split=(0.8, 0.1, 0.1)):
        super(LMClassificationSystem, self).__init__()

        input_dim = 1
        for v in X.shape[1:]:
            input_dim *= v

        self.X = X
        self.y = y
        assert sum(split) == 1.0
        self.split = split
        self.train_start = 0
        self.val_start = int(len(X) * split[0])
        self.test_start = self.val_start + int(len(X) * split[1])

        self.model = LMClassificationModel(input_dim)

    def forward(self, x):
        x = self.model(x)
        return x

    def inverse(self, coeff):
        direction = self.linear.weight * coeff
        direction = self.inn(direction, rev=True).detach().cpu()
        return direction

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.view(-1, 1))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.view(-1, 1))
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.002)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        X = self.X[:self.val_start]
        y = self.y[:self.val_start]
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=32)
        return dl

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        X = self.X[self.val_start:self.test_start]
        y = self.y[self.val_start:self.test_start]
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=32)
        return dl

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        X = self.X[self.test_start:]
        y = self.y[self.test_start:]
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=32)
        return dl


def plot_pointcloud(points, title="", block=True):
    fig = plt.figure(figsize=(5, 5))
    x, y, z = points.unbind(1)
    ax = Axes3D(fig)
    ax.scatter3D(x, -z, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(180, 90)
    ax.set_title(title)
    plt.show(block=block)


def prepare_data(data_path, target_emotion, lm3d):
    # Load data
    data = torch.load(data_path)

    # Select data
    shuffle_inds = torch.randperm(data['landmarks'].shape[0])

    # Landmarks for original labels
    emotions = data['emotions'][shuffle_inds]
    if lm3d:
        landmarks = data['lm3d'][shuffle_inds]
    else:
        landmarks = data['landmarks'][shuffle_inds]
    # Get data which corresponds to emotion and other data which doesn't
    target_emo_inds = (emotions == MAPPING[target_emotion]).nonzero()
    other_emo_inds = (emotions != MAPPING[target_emotion]).nonzero()[
        :len(target_emo_inds)]
    X = torch.cat((landmarks[target_emo_inds][:, 0],
                   landmarks[other_emo_inds][:, 0]), dim=0)
    y = torch.cat((torch.ones(len(target_emo_inds), 1),
                   torch.zeros(len(target_emo_inds), 1)), dim=0)
    # Shuffle again
    shuffle_inds2 = torch.randperm(X.shape[0])
    X = X[shuffle_inds2]
    y = y[shuffle_inds2]
    # Center landmarks
    X = (X / 127.5) - 1.

    print(f"X shape {X.shape} y shape {y.shape}")
    return X, y


def gradient_ascent(lm, model, coeff=0., n_iters=100, device='cuda'):
    coeff = torch.tensor(coeff, device=device).view(1, 1)
    # inp = lm.clone()[:, 17:].to(device)
    inp = lm.clone().to(device)
    inp.requires_grad = True

    opt = torch.optim.SGD([inp], lr=0.001)

    for i in range(n_iters):
        out = torch.sigmoid(model(inp))

        loss = F.mse_loss(out, coeff)

        opt.zero_grad()
        loss.backward()
        opt.step()

    ret = lm.clone()
    # ret[:, 17:] = inp.detach().cpu()
    ret = inp.detach().cpu()
    return ret


def train_direction_model(args):
    # Load and prepare data
    X, y = prepare_data(args.data_path, args.target_emotion, args.lm3d)

    # Train
    # system = LMClassificationSystem(X[:, 17:], y)  # Don't use chin
    system = LMClassificationSystem(X, y)
    trainer = pl.Trainer(
        gpus=1,
        logger=False,
        show_progress_bar=True,
        min_epochs=5
    )
    trainer.fit(system)
    model = system.model

    # Evaluate
    model = model.eval()
    # X_train = X[:system.val_start, 17:]
    X_train = X[:system.val_start]
    y_train = y[:system.val_start]
    # X_test = X[system.test_start:, 17:]
    X_test = X[system.test_start:]
    y_test = y[system.test_start:]
    with torch.no_grad():
        # Train set
        y_pred = torch.sigmoid(model(X_train.to(device))).cpu().numpy().reshape(-1,)
        y_pred = np.round(y_pred)
        print(f"Train Accuracy {accuracy_score(y_train.numpy(), y_pred):.4f}")

        # Test set
        y_pred = torch.sigmoid(model(X_test.to(device))).cpu().numpy().reshape(-1,)
        y_pred = np.round(y_pred)
        print(f"Test Accuracy {accuracy_score(y_test.numpy(), y_pred):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.lm3d:
        save_path = f'{args.save_dir}model_{args.target_emotion}_3d.pt'
    else:
        save_path = f'{args.save_dir}model_{args.target_emotion}.pt'
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)


def get_landmarks_image(lm):
    def draw_circle(pt, draw, radius=1):
        pt1 = tuple((pt[0] - radius, pt[1] - radius))
        pt2 = tuple((pt[0] + radius, pt[1] + radius))
        draw.ellipse([pt1, pt2], fill='white')

    # Create blank image
    img = Image.new('RGB', (256, 256))
    # Normalize landmarks
    lm_n = lm.clone()
    lm_n = (lm_n + 1.) * 127.5
    # Draw landmarks
    draw = ImageDraw.Draw(img)
    for (x, y) in lm_n:
        draw_circle((x, y), draw, radius=1)
    return img


def prepare_test_data(test_path, lm3d):
    # Load landmarks and image
    lm_path = test_path + '.landmarks3d.pt' if lm3d else test_path + '.landmarks.pt'
    lm = torch.load(lm_path).unsqueeze(0)
    img = TO_TENSOR(Image.open(test_path + '.png')).unsqueeze(0)
    img = utils.downsample_256(img)

    # Create 8 x 8 image
    img_small = transforms.Normalize([.5, .5, .5], [.5, .5, .5])(img[0]).unsqueeze(0)
    img_small = F.interpolate(img_small, size=(8, 8), mode='nearest')

    # Normalize landmarks
    lm = (lm / 127.5) - 1.

    return lm, img, img_small


def simulate_image(lm, img_small, to_latent_model, g, device='cuda'):
    # Create input
    inp = torch.cat((img_small.view(1, -1), lm.view(1, -1)), dim=1).to(device)
    # Predict latent
    latent = to_latent_model(inp)
    # Generate image
    img = g([latent], input_is_latent=True, noise=g.noises)[0].cpu()
    # Downsample to 256
    img = utils.downsample_256(img)
    # Normalize
    img = make_grid(img, normalize=True, range=(-1, 1))
    img = TO_PIL(img)
    draw = ImageDraw.Draw(img)
    # Denormalize lm
    lm_n = lm.clone()
    lm_n = (lm_n + 1.) * 127.5
    for (x, y) in lm_n[0]:
        draw.point((x, y), fill='blue')

    return img


def test_direction(args):
    # Load and prepare data
    lm, img, img_small = prepare_test_data(args.test_path, args.lm3d)
    input_dim = 1
    for v in lm.shape[1:]:
        input_dim *= v

    # Load classification model
    lm_classification = LMClassificationModel(dim=input_dim).to(args.device)
    if args.lm3d:
        model_path = f'{args.save_dir}model_{args.target_emotion}_3d.pt'
    else:
        model_path = f'{args.save_dir}model_{args.target_emotion}.pt'
    print(model_path)
    lm_classification.load_state_dict(torch.load(model_path))

    # Positive direction
    lm_pos = gradient_ascent(lm, lm_classification, 4., n_iters=500)

    if args.lm3d:
        plot_pointcloud(lm[0], 'Original', block=False)
        plot_pointcloud(lm_pos[0], f'direction {args.target_emotion}')
    else:
        # Load landmarks to latent model
        to_latent_model = models.lmToStyleGANLatent().eval().to(device)
        state_dict = torch.load('saves/pre-trained/lmToStyleGANLatent_ravdess.pt')
        to_latent_model.load_state_dict(state_dict)

        # Init StyleGAN generator
        g = style_gan_2.PretrainedGenerator1024().eval().to(device)

        # Original
        lm_img = get_landmarks_image(lm[0])
        img = simulate_image(lm, img_small, to_latent_model, g)
        # Positive direction
        lm_img_pos = get_landmarks_image(lm_pos[0])
        img_pos = simulate_image(lm_pos, img_small, to_latent_model, g)

        save_tensor1 = torch.stack((TO_TENSOR(img), TO_TENSOR(img_pos)), dim=0)
        save_tensor2 = torch.stack((TO_TENSOR(lm_img), TO_TENSOR(lm_img_pos)), dim=0)

        save_image(save_tensor1, f"{args.save_dir}image_{args.target_emotion}.png")
        save_image(save_tensor2, f"{args.save_dir}lm_image_{args.target_emotion}.png")


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--target_emotion', type=str, default='happy')
    parser.add_argument('--save_dir', type=str, default='saves/landmarks_emotion_direction')
    parser.add_argument('--lm3d', type=bool, default=True)
    parser.add_argument(
        '--test_path', type=str, default='/home/meissen/Datasets/RAVDESS/Aligned256/Actor_01/01-01-01-01-01-01-01/005')
    parser.add_argument(
        '--data_path', type=str, default='/home/meissen/Datasets/RAVDESS/Aligned256/latent_data.pt')
    args = parser.parse_args()

    # Correct save_path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Perform action
    if args.train:
        train_direction_model(args)
    elif args.test:
        test_direction(args)
