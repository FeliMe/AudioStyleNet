import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

from datetime import datetime
from lpips import PerceptualLoss
from matplotlib.widgets import Slider, Button
from my_models import style_gan_2
from my_models import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import datasets, utils
from torchvision import transforms
from torchvision.utils import save_image, make_grid


HOME = os.path.expanduser('~')


class solverEncoder:
    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.args = args

        self.initial_lr = self.args.lr
        self.lr = self.args.lr
        self.lr_rampdown_length = 0.3
        self.lr_rampup_length = 0.1

        # Load generator
        self.g = style_gan_2.PretrainedGenerator1024().eval().to(device)
        for param in self.g.parameters():
            param.requires_grad = False

        # Init global step
        self.global_step = 0

        # Define encoder model
        self.e = models.resNetAdaINEncoder(
            args.n_factors, args.n_latent, args.len_dataset
        ).train().to(self.device)
        print(f"Len factor db {args.len_dataset}")

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
        if not self.args.debug and not self.args.test:
            tb_dir = self.args.save_dir
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

    def get_batch(self, data_loaders, evaluate=False):
        if random.random() > self.args.labeled_probability or evaluate:
            batch = next(iter(data_loaders['unlabeled']))
            img = batch['img'].to(self.device)
            factor = None
            index = batch['index'].to(self.device)
        else:
            batch = next(iter(data_loaders['labeled']))
            img = batch['img'].to(self.device)
            factor = batch['y'].to(self.device)
            index = None
        return img, factor, index

    def forward(self, img, factor, index, evaluation=False):
        # Encode
        if evaluation:
            self.e.eval()
        latent_offset = self.e(img, factor, index)
        if evaluation:
            self.e.train()
        # Add mean (we only want to compute offset to mean latent)
        latent = latent_offset + self.g.latent_avg
        # Decode
        gen = self.g([latent], input_is_latent=True, noise=self.g.noises)[0]
        # Downsample to 256 x 256
        gen = utils.downsample_256(gen)
        # Compute perceptual loss
        loss = self.criterion(gen, img).mean()

        return loss, gen

    def train(self, n_iters, train_loaders):
        print("Start training")

        def clip_factors(factors):
            f = factors.data
            f.sub_(torch.min(f)).div_(torch.max(f) - torch.min(f))

        pbar = tqdm(total=n_iters, initial=self.global_step)
        while self.global_step < n_iters:
            img, factor, index = self.get_batch(train_loaders, evaluate=False)

            # Update learning rate
            t = self.global_step / n_iters
            self.update_lr(t)

            loss, gen = self.forward(img, factor, index)

            # Optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # Normalize factors to [0., 1.]
            clip_factors(self.e.factors)

            self.global_step += 1
            pbar.update()

            # Update progress bar
            pbar.set_description('Step {gs} - '
                                 'Train loss {tl:.4f} - '
                                 'Max mem {vl:.2f} GB - '
                                 'lr {lr:.4f}'.format(
                                     gs=self.global_step,
                                     tl=loss,
                                     vl=torch.cuda.max_memory_allocated(self.device) / 1e9,
                                     lr=self.lr
                                 ))

            if not self.args.debug:
                if self.global_step % self.args.log_train_every == 0:
                    self.writer.add_scalars(
                        'loss', {'train': loss}, self.global_step)

                if self.global_step % self.args.save_every == 0:
                    self.save()

                if self.global_step % self.args.save_img_every == 0:
                    # Save train sample
                    save_tensor = torch.cat(
                        (img.detach(), gen.detach().clamp(-1., 1.)), dim=0)
                    save_image(
                        save_tensor,
                        f'{self.args.save_dir}train_gen_{self.global_step}.png',
                        normalize=True,
                        range=(-1, 1),
                        nrow=min(8, self.args.batch_size)
                    )

        self.save()
        print('Done.')

    def test_model(self, test_image_path):
        """
        Matplotlib slideshow
        """

        def prepare_display(img):
            im = make_grid(img.clone(), normalize=True, range=(-1, 1)).cpu()
            return transforms.ToPILImage('RGB')(im)

        # Prepare inputs
        t = transforms.Compose([
            transforms.ToTensor(),
            datasets.Downsample(256),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
        test_img = t(Image.open(test_image_path)).unsqueeze(0).to(self.device)

        # Set up plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.55)
        print(test_img.min(), test_img.max(), test_img.mean())
        im = plt.imshow(prepare_display(test_img))
        print(test_img.min(), test_img.max(), test_img.mean())
        plt.axis('off')
        ax_sliders = [plt.axes([0.2, 0.1 + 0.05 * i, 0.65, 0.03],
                               facecolor='lightgoldenrodyellow') for i in range(self.args.n_factors)]
        sliders = [Slider(ax_slider, emotion, 0.0, 1.0, valinit=0, valstep=0.01)
                   for emotion, ax_slider in zip(self.args.emotions, ax_sliders)]

        self.e.eval()

        def update(val):
            factor = torch.tensor([slider.val for slider in sliders]).view(
                1, -1).to(self.device)
            with torch.no_grad():
                # Encode
                latent_offset = self.e(test_img, factor, None)
                latent = latent_offset + self.g.latent_avg

                # Decode
                gen = self.g(
                    [latent], input_is_latent=True, noise=self.g.noises)[0]

                # Downsample to 256 x 256
                gen = utils.downsample_256(gen)

            # Update plot
            im.set_data(prepare_display(gen))
            fig.canvas.draw_idle()

        for slider in sliders:
            slider.on_changed(update)

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color='lightgoldenrodyellow',
                        hovercolor='0.975')

        def reset(event):
            for slider in sliders:
                slider.reset()

        button.on_clicked(reset)
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
    # Flags
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')

    # Hparams
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--labeled_probability', type=float, default=0.05)
    parser.add_argument('--n_latent', type=int, default=32)

    parser.add_argument('--emotions', type=list, default=['neutral', 'calm', 'happy', 'sad', 'angry',
                                                          'fearful', 'disgust', 'surprised'])

    # Logging params
    parser.add_argument('--n_iters', type=int, default=40000)
    parser.add_argument('--log_train_every', type=int, default=1)
    parser.add_argument('--log_val_every', type=int, default=1000)
    parser.add_argument('--save_img_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=4000)
    parser.add_argument('--save_dir', type=str,
                        default='saves/semi_supervised_encoder/')

    # Test params
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test_img_path', type=str, default=None)
    args = parser.parse_args()
    args.n_factors = len(args.emotions)

    if args.cont or args.test:
        assert args.model_path is not None

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

    # Data loading
    rav_paths, _, _ = datasets.ravdess_get_paths(
        root_path=HOME + "/Datasets/RAVDESS/Aligned256/",
        flat=True,
        validation_split=0.0,
        emotions=args.emotions
    )
    ffhq_paths, _ = datasets.ffhq_get_paths(
        root_path=HOME + '/Datasets/FFHQ/Aligned256/',
        train_split=1.0
    )
    rav_ds = datasets.RAVDESSFlatDataset(
        paths=rav_paths,
        device=device,
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5],
        label_one_hot=True
    )
    ffhq_ds = datasets.FFHQDataset(
        paths=ffhq_paths,
        normalize=True,
        mean=[.5, .5, .5],
        std=[.5, .5, .5]
    )
    train_loaders = {
        'labeled': torch.utils.data.DataLoader(
            rav_ds, batch_size=args.batch_size, shuffle=True),
        'unlabeled': torch.utils.data.DataLoader(
            ffhq_ds, batch_size=args.batch_size, shuffle=True)
    }
    args.len_dataset = len(ffhq_ds)

    # Init solver
    solver = solverEncoder(args)

    # Train
    if args.test:
        solver.test_model(args.test_img_path)
    else:
        solver.train(args.n_iters, train_loaders)
