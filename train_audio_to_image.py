import argparse
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from datetime import datetime
from glob import glob
from my_models import models, style_gan_2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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

        # Init global step
        self.global_step = 0
        self.step_start = 0

        # Init Generator
        self.g = style_gan_2.PretrainedGenerator1024().eval().to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

        # Define audio encoder
        self.audio_encoder = models.AudioExpressionNet3(args.T).to(self.device).train()

        if self.args.cont or self.args.test:
            path = self.args.model_path
            self.audio_encoder.load_state_dict(torch.load(path))
            self.global_step = int(path.split(
                '/')[-1].split('.')[0].split('model')[-1])
            self.step_start = int(path.split(
                '/')[-1].split('.')[0].split('model')[-1])

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.audio_encoder),
            utils.count_trainable_params(self.audio_encoder)
        ))

        # Select optimizer and loss criterion
        self.opt = torch.optim.Adam(self.audio_encoder.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction='none')

        # Loss weighting vector
        self.loss_weight = torch.load(
            'saves/pre-trained/latent_4to8_std_avg.pt').unsqueeze(0).to(self.device)

        # Set up tensorboard
        if not self.args.debug and not self.args.test:
            tb_dir = self.args.save_dir
            self.writer = SummaryWriter(tb_dir)
            print(f"Logging run to {tb_dir}")

            # Create save dir
            os.makedirs(self.args.save_dir + 'models', exist_ok=True)
            os.makedirs(self.args.save_dir + 'sample', exist_ok=True)

    def save(self):
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        print(f"Saving: {save_path}")
        torch.save(self.audio_encoder.state_dict(), save_path)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.opt.param_groups[0]['lr'] = self.lr

    def unpack_data(self, batch):
        audio = batch['audio'].to(self.device)
        # audio = batch['audio'][:, self.args.T // 2].to(self.device)
        input_latent = batch['input_latent'].to(self.device)
        target_latent = batch['target_latent'].to(self.device)
        return audio, input_latent, target_latent

    def forward(self, audio, input_latent):
        latent_offset = self.audio_encoder(audio, input_latent[:, 4:8])
        # latent_offset = self.audio_encoder(audio, input_latent)
        # Add mean (we only want to compute offset to mean latent)
        prediction = input_latent
        prediction[:, 4:8] += latent_offset
        # prediction = input_latent + latent_offset

        return prediction

    def get_loss(self, pred, target, validate=False):
        loss = self.criterion(pred[:, 4:8], target[:, 4:8])
        # loss = self.criterion(pred, target)

        # Weight individual loss components
        loss = loss * self.loss_weight

        # Mean loss
        loss = loss.mean()

        return loss

    def train(self, data_loaders, n_iters):
        print("Start training")
        pbar = tqdm(total=n_iters)
        i_iter = 0
        avg_train_loss = 0.
        pbar_avg_train_loss = 0.
        val_loss = 0.

        while i_iter < n_iters:
            for batch in data_loaders['train']:
                # Unpack batch
                audio, input_latent, target = self.unpack_data(batch)

                # Update learning rate
                # t = i_iter / n_iters
                # self.update_lr(t)

                # Encode
                pred = self.forward(audio, input_latent)

                # Compute perceptual loss
                loss = self.get_loss(pred, target, validate=False)

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                avg_train_loss += loss.item()
                pbar_avg_train_loss += loss.item()

                self.global_step += 1
                i_iter += 1
                pbar.update()

                if self.global_step % self.args.log_val_every == 0:
                    val_loss = self.validate(data_loaders)

                if self.global_step % self.args.update_pbar_every == 0:
                    pbar_avg_train_loss /= self.args.update_pbar_every
                    pbar.set_description('step [{gs}/{ni}] - '
                                         'train loss {tl:.4f} - '
                                         'val loss {vl:.4f} - '
                                         'lr {lr}'.format(
                                             gs=self.global_step,
                                             ni=n_iters,
                                             tl=pbar_avg_train_loss,
                                             vl=val_loss,
                                             lr=self.lr
                                         ))
                    pbar_avg_train_loss = 0.
                    print("")

                # Logging and evaluating
                if not self.args.debug:
                    if self.global_step % self.args.log_train_every == 0:
                        avg_train_loss /= max(1, float(self.args.log_train_every))
                        self.writer.add_scalars('loss', {'train': avg_train_loss}, self.global_step)
                        avg_train_loss = 0.

                    if self.global_step % self.args.log_val_every == 0:
                        self.writer.add_scalars('loss', {'val': val_loss}, self.global_step)

                    if self.global_step % self.args.save_every == 0:
                        self.save()

                    if self.global_step % self.args.eval_every == 0:
                        self.eval(data_loaders['train'], f'train_gen_{self.global_step}.png')
                        self.eval(data_loaders['val'], f'val_gen_{self.global_step}.png')

                # Break if n_iters is reached and still in epoch
                if i_iter == n_iters:
                    break

        self.save()
        print('Done.')

    def validate(self, data_loaders):
        val_loss = 0.
        for batch in data_loaders['val']:
            # Unpack batch
            audio, input_latent, target = self.unpack_data(batch)

            with torch.no_grad():
                # Forward
                pred = self.forward(audio, input_latent)

            val_loss += self.get_loss(pred, target, validate=True)
        return val_loss / float(len(data_loaders['val']))

    def eval(self, data_loader, sample_name):
        # Train sample
        batch = next(iter(data_loader))
        # Unpack batch
        audio, input_latent, target = self.unpack_data(batch)

        n_display = min(4, self.args.batch_size)
        audio = audio[:n_display]
        target = target[:n_display]
        input_latent = input_latent[:n_display]

        with torch.no_grad():
            # Forward
            pred = self.forward(audio, input_latent.clone())
            input_img, _ = self.g([input_latent], input_is_latent=True, noise=self.g.noises)
            input_img = utils.downsample_256(input_img)

            pred, _ = self.g(
                [pred], input_is_latent=True, noise=self.g.noises)
            pred = utils.downsample_256(pred)
            target_img, _ = self.g(
                [target], input_is_latent=True, noise=self.g.noises)
            target_img = utils.downsample_256(target_img)

        # Normalize images to display
        input_img = make_grid(input_img, normalize=True, range=(-1, 1))
        pred = make_grid(pred, normalize=True, range=(-1, 1))
        target_img = make_grid(target_img, normalize=True, range=(-1, 1))
        diff = (target_img - pred) * 5

        img_tensor = torch.stack((pred, target_img, diff, input_img), dim=0)
        save_image(
            img_tensor,
            f'{self.args.save_dir}sample/{sample_name}',
            nrow=1
        )

        # Val sample

    def test_model(self, test_latent_path, test_sentence_path):
        self.audio_encoder.eval()
        test_latent = torch.load(test_latent_path).unsqueeze(0).to(self.device)

        audio_paths = sorted(glob(test_sentence_path + '*.deepspeech.npy'))[:100]
        audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32) for p in audio_paths]).to(self.device)
        pad = self.args.T // 2
        audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
        audios = audios.unfold(0, self.args.T, 1).permute(0, 3, 1, 2)

        target_latent_paths = sorted(glob(test_sentence_path + '*.latent.pt'))[:100]
        target_latents = torch.stack([torch.load(p) for p in target_latent_paths]).to(self.device)

        pbar = tqdm(total=len(target_latents))
        avg_loss = 0.

        tmp_dir = self.args.save_dir + '.temp/'
        os.makedirs(tmp_dir, exist_ok=True)
        for i, (audio, target_latent) in enumerate(zip(audios, target_latents)):
            audio = audio.unsqueeze(0)
            target_latent = target_latent.unsqueeze(0)
            with torch.no_grad():
                input_latent = test_latent.clone()
                latent = self.forward(audio, input_latent)
                # Generate images
                pred = self.g([latent], input_is_latent=True, noise=self.g.noises)[0]
                target = self.g([target_latent], input_is_latent=True, noise=self.g.noises)[0]
                # Get loss
                loss = self.get_loss(latent, target_latent, validate=True)
                avg_loss += loss.item()
            pbar.update()
            pbar.set_description(f"Loss {loss.item():.4f}")
            # Downsample
            pred = utils.downsample_256(pred).cpu()
            target = utils.downsample_256(target).cpu()
            # Normalize
            pred = make_grid(pred, normalize=True, range=(-1, 1))
            target = make_grid(target, normalize=True, range=(-1, 1))
            diff = (target - pred) * 5

            save_tensor = torch.stack((pred, target, diff), dim=0)
            save_image(save_tensor, f"{tmp_dir}{str(i + 1).zfill(5)}.png")

        # Convert output frames to video
        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        os.system(
            f'ffmpeg -framerate 25 -i %05d.png -c:v libx264 -r 25 -pix_fmt yuv420p ../out.mp4')

        # Remove generated frames and keep only video
        os.chdir(original_dir)
        os.system(f'rm -r {tmp_dir}')

        print(f"Average loss {avg_loss / len(target_latents):.4f}")

        self.audio_encoder.train()


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)

    # Hparams
    parser.add_argument('--lambda_l1', type=float, default=10.)
    parser.add_argument('--batch_size', type=int, default=32)  # 128
    parser.add_argument('--lr', type=int, default=0.0002)  # 0.0002
    parser.add_argument('--T', type=int, default=8)

    # Logging args
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--update_pbar_every', type=int, default=1000)
    parser.add_argument('--log_train_every', type=int, default=1000)
    parser.add_argument('--log_val_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=100000)
    parser.add_argument('--eval_every', type=int, default=100000)
    parser.add_argument('--save_dir', type=str, default='saves/audio_encoder/')

    # Path args
    parser.add_argument('--data_path', type=str, default='/home/meissen/Datasets/AudioDataset/Aligned256/')
    parser.add_argument('--test_latent', type=str, default='saves/projected_images/obama.pt')
    parser.add_argument('--test_sentence', type=str,
                        default='/home/meissen/Datasets/Tagesschau/test_sentence_trump_deepspeech/')
    args = parser.parse_args()

    if args.cont or args.test:
        assert args.model_path is not None
        assert args.test_latent is not None

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    args.save_dir += timestamp

    if args.cont or args.test:
        args.save_dir = '/'.join(args.model_path.split('/')[:-2]) + '/'

    if args.debug:
        print("DEBUG MODE. NO LOGGING")
    else:
        print("Saving run to {}".format(args.save_dir))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # Load data
    train_paths, val_paths = datasets.tagesschau_get_paths(
        args.data_path, 0.9, max_frames_per_vid=1000)

    if args.overfit:
        train_paths = [train_paths[0]]
        val_paths = train_paths
        print(f"OVERFITTING ON {train_paths[0][0]}")

    print(f"Sample path training {train_paths[0][0]}")
    print(f"Sample path validation {val_paths[0][0]}")

    train_ds = datasets.TagesschauAudioDataset(
        paths=train_paths,
        load_img=False,
        load_latent=True,
        T=args.T,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    val_ds = datasets.TagesschauAudioDataset(
        paths=val_paths,
        load_img=False,
        load_latent=True,
        T=args.T,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    train_sampler = datasets.RandomTagesschauAudioSampler(
        train_paths, args.T, args.batch_size, 10000, weighted=True)
    val_sampler = datasets.RandomTagesschauAudioSampler(
        val_paths, args.T, args.batch_size, 20, weighted=True)

    print(f"Dataset length: Train {len(train_ds)} val {len(val_ds)}")
    data_loaders = {
        'train': DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        ),
        'val': DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    }

    # Init solver
    solver = Solver(args)

    # Train
    if args.test:
        solver.test_model(args.test_latent, args.test_sentence)
    else:
        solver.train(data_loaders, args.n_iters)
        print("Finished training.")
