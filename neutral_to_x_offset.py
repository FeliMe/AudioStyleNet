import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch

from datetime import datetime
from lpips import PerceptualLoss
from matplotlib.widgets import Slider
from my_models.style_gan_2 import Generator
from my_models import models
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from utils import datasets, utils


HOME = os.path.expanduser('~')


class Solver:
    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.args = args

        self.initial_lr = self.args.lr
        self.lr = self.args.lr
        self.lr_rampdown_length = 0.4
        self.lr_rampup_length = 0.1

        # Load generator
        self.g = Generator(
            1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        for param in self.g.parameters():
            param.requires_grad = False
        self.latent_avg = self.g.latent_avg.repeat(
            18, 1).unsqueeze(0).to(self.device)

        # Init global step
        self.global_step = 0

        # Define encoder model
        self.e = models.resNetOffsetEncoder(1).to(self.device).train()
        # print(self.e)

        if self.args.cont or self.args.test:
            path = self.args.model_path
            self.e.load_state_dict(torch.load(path))
            self.global_step = int(path.split(
                '/')[-1].split('.')[0].split('model')[-1])

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.e),
            utils.count_trainable_params(self.e)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(self.e.parameters(), lr=self.initial_lr)
        self.criterion = PerceptualLoss(
            model='net-lin', net='vgg').to(self.device)

        # Set up tensorboard
        if self.args.log and not self.args.test:
            self.args.save_dir.split('/')[-1]
            tb_dir = 'tensorboard_runs/neutral_to_x/' + \
                self.args.save_dir.split('/')[-2]
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

        # Create save dir
        os.makedirs(self.args.save_dir + 'models', exist_ok=True)

    def save(self):
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.e.state_dict(), save_path)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt.param_groups[0]['lr'] = self.lr

    def train(self, data_loaders, n_epochs):
        print("Start training")
        val_loss = 0.
        n_iters = self.global_step + (n_epochs * len(data_loaders['train']))
        pbar = tqdm(range(n_epochs))
        for i_epoch in pbar:
            for _, batch in enumerate(data_loaders['train']):
                # Unpack batch
                img_n = batch['neutral'].to(device)
                img_x = batch['x'].to(device)
                score_x = batch['score_x'].to(device)

                # Update learning rate
                t = self.global_step / n_iters
                self.update_lr(t)

                # Encode
                latent_offset, offset_to_x = self.e(img_n, score_x)
                # Add mean (we only want to compute offset to mean latent)
                latent_n = latent_offset + self.latent_avg
                latent_x = latent_n + offset_to_x

                # Decode
                img_n_gen, _ = self.g(
                    [latent_n], input_is_latent=True, noise=self.g.noises)
                img_x_gen, _ = self.g(
                    [latent_x], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_n_gen = utils.downsample_256(img_n_gen)
                img_x_gen = utils.downsample_256(img_x_gen)

                # Compute perceptual loss
                loss_n = self.criterion(img_n_gen, img_n).mean()
                loss_x = self.criterion(img_x_gen, img_x).mean()

                loss = 0.5 * (loss_x + loss_n)

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    pbar.set_description('step [{gs}/{ni}] - '
                                         'train loss {tl:.4f} - '
                                         'val loss {vl:.4f} - '
                                         'lr {lr:.4f}'.format(
                                             gs=self.global_step,
                                             ni=n_iters,
                                             tl=loss,
                                             vl=val_loss,
                                             lr=self.lr
                                         ))
                    if self.args.log:
                        self.writer.add_scalar(
                            'loss/train', loss, self.global_step)

                if self.global_step % self.args.save_every == 0:
                    self.save()

                if self.global_step % self.args.eval_every == 0:
                    # Save train sample
                    save_tensor = torch.cat(
                        (img_n.detach(), img_x.detach(), img_n_gen.detach().clamp(-1., 1.), img_x_gen.detach().clamp(-1., 1.)),
                        dim=0
                    )
                    save_image(
                        save_tensor,
                        f'{self.args.save_dir}train_gen_{self.global_step}.png',
                        normalize=True,
                        range=(-1, 1),
                        nrow=min(8, self.args.batch_size)
                    )

                    # Eval one batch
                    val_loss = self.eval(data_loaders['val'])
                    print("")

        self.save()
        print('Done.')

    def eval(self, val_loader):
        # Set encoder to eval
        self.e.eval()

        # Get random validation batch
        batch = next(iter(val_loader))

        # Unpack batch
        img_n = batch['neutral'].to(device)
        img_x = batch['x'].to(device)
        score_x = batch['score_x'].to(device)

        # Encode
        with torch.no_grad():
            # Encode
            latent_offset, offset_to_x = self.e(img_n, score_x)
            # Add mean (we only want to compute offset to mean latent)
            latent_n = latent_offset + self.latent_avg
            latent_x = latent_n + offset_to_x

            # Decode
            img_n_gen, _ = self.g(
                [latent_n], input_is_latent=True, noise=self.g.noises)
            img_x_gen, _ = self.g(
                [latent_x], input_is_latent=True, noise=self.g.noises)

            # Downsample to 256 x 256
            img_n_gen = utils.downsample_256(img_n_gen)
            img_x_gen = utils.downsample_256(img_x_gen)

            # Compute perceptual loss
            loss_n = self.criterion(img_n_gen, img_n).mean()
            loss_x = self.criterion(img_x_gen, img_x).mean()

            val_loss = 0.5 * (loss_x + loss_n)

            if self.args.log:
                self.writer.add_scalar(
                    'loss/val', val_loss, self.global_step)

        # Save val sample
        save_tensor = torch.cat(
            (img_n.detach(), img_x.detach(), img_n_gen.detach().clamp(-1., 1.), img_x_gen.detach().clamp(-1., 1.)),
            dim=0
        )
        save_image(
            save_tensor,
            f'{self.args.save_dir}val_gen_{self.global_step}.png',
            normalize=True,
            range=(-1, 1),
            nrow=min(8, self.args.batch_size)
        )

        # Set encoder back to train
        self.e.train()

        return val_loss

    def test_model(self, test_latent_path, n_img=8):
        test_latent = torch.load(test_latent_path).to(self.device)
        sample = next(iter(data_loaders['val']))
        img_n = sample['neutral'][0].unsqueeze(0).to(self.device)
        scores = torch.tensor(np.linspace(0., 1., n_img),
                              dtype=torch.float32, device=device)

        test_img, _ = self.g(
            [test_latent], input_is_latent=True, noise=self.g.noises
        )
        test_img = utils.downsample_256(test_img)

        imgs_test = [test_img]
        imgs_train_actor = [utils.downsample_256(img_n)]
        self.e.eval()
        for score in scores:
            # Encode
            print(f"Score: {score.item():.4f}")
            with torch.no_grad():
                latent_offset, offset_to_x = self.e(img_n, score.view((1, -1)))
                # Add mean (we only want to compute offset to mean latent)
                latent_n = latent_offset + self.latent_avg + offset_to_x
                latent_x = test_latent + offset_to_x

                # Decode
                img_n_gen, _ = self.g(
                    [latent_n], input_is_latent=True, noise=self.g.noises)
                img_x_gen, _ = self.g(
                    [latent_x], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_n_gen = utils.downsample_256(img_n_gen)
                img_x_gen = utils.downsample_256(img_x_gen)

            imgs_test.append(img_x_gen)
            imgs_train_actor.append(img_n_gen)

        imgs_test = torch.cat(imgs_test, dim=0)
        imgs_train_actor = torch.cat(imgs_train_actor, dim=0)

        tmp_path = "{}{}_offset_{}.png".format(
            self.args.save_dir,
            test_latent_path.split('/')[-1].split('.')[0],
            self.args.emotion
        )
        print("Saving to {}".format(tmp_path))
        save_image(
            imgs_test,
            tmp_path,
            normalize=True,
            range=(-1, 1),
            nrow=n_img + 1,
        )
        save_image(
            imgs_train_actor,
            f'{self.args.save_dir}test_scores.png',
            normalize=True,
            range=(-1, 1),
            nrow=n_img + 1,
        )
        self.e.train()

    def test_model2(self, test_latent_path):
        """
        Matplotlib slideshow
        """
        # Prepare inputs
        test_latent = torch.load(test_latent_path).to(self.device)
        sample = next(iter(data_loaders['val']))
        img_n = sample['neutral'][0].unsqueeze(0).to(self.device)

        test_img, _ = self.g(
            [test_latent], input_is_latent=True, noise=self.g.noises
        )
        test_img = utils.downsample_256(test_img)
        test_img = make_grid([test_img], normalize=True)
        t = transforms.ToPILImage(mode='RGB')

        # Set up plot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        im = plt.imshow(t(test_img[0].cpu()))
        plt.axis('off')
        # ax.margins(x=0)
        ax_slider1 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider1 = Slider(ax_slider1, 'score', 0.0, 1.0, valinit=0.0, valstep=0.1)

        self.e.eval()

        def update(val):
            score = torch.tensor(slider1.val).view(1, -1).to(self.device)
            # Encode
            print(f"Score: {score.item():.4f}")
            with torch.no_grad():
                _, offset_to_x = self.e(img_n, score)
                latent_x = test_latent + offset_to_x

                # Decode
                img_x_gen, _ = self.g(
                    [latent_x], input_is_latent=True, noise=self.g.noises)

                # Downsample to 256 x 256
                img_x_gen = utils.downsample_256(img_x_gen)

            # Update plot
            img_x_gen = make_grid([img_x_gen], normalize=True)
            im.set_data(t(img_x_gen[0].cpu()))
            fig.canvas.draw_idle()

        slider1.on_changed(update)
        plt.show()

        self.e.train()


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emotion', type=str, required=True)
    parser.add_argument('--lin_scores', action='store_true')
    parser.add_argument('--score_type', type=str, default='fer', help="'fer' or 'ravdess'")
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test_latent', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='saves/neutral_to_x/')
    args = parser.parse_args()

    if args.cont or args.test:
        assert args.model_path is not None
        assert args.test_latent is not None

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    if args.cont or args.test:
        args.save_dir = '/'.join(args.model_path.split('/')[:-2]) + '/'

    print("Saving run to {}".format(args.save_dir))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Load data
    train_paths, val_paths, all_paths = datasets.get_paths(
        HOME + '/Datasets/RAVDESS/Aligned256/',
        validation_split=0.25,
        # actors=[1],
        flat=False,
    )
    train_ds = datasets.RAVDESSNeutralToXDataset(
        train_paths,
        all_paths,
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
        emotion_x=args.emotion,
        lin_scores=args.lin_scores,
        score_type=args.score_type,
    )
    val_ds = datasets.RAVDESSNeutralToXDataset(
        val_paths,
        all_paths,
        device=device,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
        emotion_x=args.emotion,
        lin_scores=args.lin_scores,
        score_type=args.score_type,
    )
    data_loaders, dataset_sizes = datasets.get_data_loaders(
        train_ds, val_ds, batch_size=args.batch_size, use_cuda=True, val_batch_size=1)
    sample = next(iter(data_loaders['train']))

    # Init solver
    solver = Solver(args)

    # Train
    if args.test:
        solver.test_model2(args.test_latent)
    else:
        solver.train(data_loaders, args.n_epochs)
