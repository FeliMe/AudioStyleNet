import argparse
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from datetime import datetime
from glob import glob
from lpips import PerceptualLoss
from my_models import models, style_gan_2
from subprocess import Popen, PIPE
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from utils import datasets, utils


HOME = os.path.expanduser('~')
RAIDROOT = os.environ['RAIDROOT']
DATAROOT = os.environ['DATAROOT']


class Solver:
    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.args = args

        self.initial_lr = self.args.lr
        self.lr = self.args.lr
        self.lr_rampdown_length = 0.4

        # Init global step
        self.global_step = 0
        self.step_start = 0

        # Init Generator
        self.g = style_gan_2.PretrainedGenerator1024().eval().to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

        # Define audio encoder
        self.audio_encoder = models.AudioExpressionNet3(args.T).to(self.device).train()

        # Print # parameters
        print("# params {} (trainable {})".format(
            utils.count_params(self.audio_encoder),
            utils.count_trainable_params(self.audio_encoder)
        ))

        # Select optimizer and loss criterion
        self.optim = torch.optim.Adam(self.audio_encoder.parameters(), lr=self.lr)
        self.lpips = PerceptualLoss(model='net-lin', net='vgg', gpu_id=args.gpu)

        if self.args.cont or self.args.test:
            path = self.args.model_path
            self.load(path)
            self.step_start = self.global_step

        # Mouth mask for image
        mouth_mask = torch.load('saves/pre-trained/tagesschau_mouth_mask_5std.pt').to(self.device)
        # eyes_mask = torch.load('saves/pre-trained/tagesschau_eyes_mask_3std.pt').to(self.device)
        self.image_mask = mouth_mask.clamp(0., 1.)
        # self.image_mask = (mouth_mask + eyes_mask).clamp(0., 1.)

        # MSE mask
        self.mse_mask = torch.load('saves/pre-trained/mse_mask_var+1.pt')[4:8].unsqueeze(0).to(self.device)

        # Set up tensorboard
        if not self.args.debug and not self.args.test:
            tb_dir = self.args.save_dir
            # self.writer = SummaryWriter(tb_dir)
            self.train_writer = utils.HparamWriter(tb_dir + 'train/')
            self.val_writer = utils.HparamWriter(tb_dir + 'val/')
            self.train_writer.log_hyperparams(self.args)
            print(f"Logging run to {tb_dir}")

            # Create save dir
            os.makedirs(self.args.save_dir + 'models', exist_ok=True)
            os.makedirs(self.args.save_dir + 'sample', exist_ok=True)

    def about_time(self, condition):
        return self.global_step % condition == 0

    def save(self):
        save_path = f"{self.args.save_dir}models/model{self.global_step}.pt"
        torch.save({
            'model': self.audio_encoder.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'global_step': self.global_step,
        }, save_path)
        print(f"Saving: {save_path}")

    def load(self, path):
        print(f"Loading audio_encoder weights from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        if type(checkpoint) == dict:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.audio_encoder.load_state_dict(checkpoint['model'])
            self.global_step = checkpoint['global_step']
        else:
            self.audio_encoder.load_state_dict(checkpoint)

    def update_lr(self, t):
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        # lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        self.lr = self.initial_lr * lr_ramp
        self.optim.param_groups[0]['lr'] = self.lr

    def unpack_data(self, batch):
        audio = batch['audio'].to(self.device)
        input_latent = batch['input_latent'].to(self.device)
        target_latent = batch['target_latent'].to(self.device)
        target_img = batch['target_img'].to(self.device)

        aux_input = input_latent[:, 4:8]

        return audio, input_latent, aux_input, target_latent, target_img

    def forward(self, audio, input_latent, aux_input):
        latent_offset = self.audio_encoder(audio, aux_input)
        prediction = input_latent.clone()
        if not self.audio_encoder.training:
            latent_offset *= self.args.test_multiplier
            prediction[:, 4:8] += latent_offset
            # Truncation trick
            prediction[:, 4:8] = self.g.latent_avg + self.args.test_truncation * \
                (prediction[:, 4:8] - self.g.latent_avg)
        else:
            prediction[:, 4:8] += latent_offset

        return prediction

    def get_loss(self, pred, target_latent, target_image, validate=False):
        latent_mse = F.mse_loss(pred[:, 4:8], target_latent[:, 4:8], reduction='none')
        latent_mse *= self.mse_mask
        latent_mse = latent_mse.mean()

        # Reconstruct image
        pred_img = self.g([pred], input_is_latent=True, noise=self.g.noises)[0]
        pred_img = utils.downsample_256(pred_img)

        # Visualize
        # from torchvision import transforms
        # sample_pred = make_grid(pred_img[0].cpu(), normalize=True, range=(-1, 1))
        # sample_target = make_grid(target_image[0].cpu(), normalize=True, range=(-1, 1))
        # sample_pred_masked = sample_pred * self.image_mask.cpu()
        # sample_target_masked = sample_target * self.image_mask.cpu()
        # print(self.image_mask.min(), self.image_mask.max())
        # print(sample_pred.shape, self.image_mask.shape, sample_pred_masked.shape)
        # print(sample_target.shape, self.image_mask.shape, sample_target_masked.shape)
        # transforms.ToPILImage('RGB')(sample_pred).show()
        # transforms.ToPILImage('RGB')(sample_target).show()
        # transforms.ToPILImage('RGB')(sample_pred_masked).show()
        # transforms.ToPILImage('RGB')(sample_target_masked).show()
        # 1 / 0

        # Image loss
        if self.args.image_loss_type == 'lpips':
            l1_loss = self.lpips(pred_img * self.image_mask, target_image * self.image_mask).mean()
        elif self.args.image_loss_type == 'l1':
            l1_loss = F.l1_loss(pred_img, target_image, reduction='none')
            l1_loss *= self.image_mask
            l1_loss = l1_loss.sum() / self.image_mask.sum()
        else:
            raise NotImplementedError

        loss = self.args.latent_loss_weight * latent_mse + self.args.photometric_loss_weight * l1_loss

        # print(f"Loss {loss.item():.4f}, latent_mse {latent_mse.item() * self.args.latent_loss_weight:.4f}, image_l1 {l1_loss.item() * self.args.photometric_loss_weight:.4f}")
        return {'loss': loss, 'latent_mse': latent_mse, 'image_l1': l1_loss}

    @staticmethod
    def _reset_loss_dict(loss_dict):
        for key in loss_dict.keys():
            loss_dict[key] = 0.
        return loss_dict

    def train(self, data_loaders, n_iters):
        print("Start training")
        pbar = tqdm(total=n_iters)
        i_iter = 0
        pbar_avg_train_loss = 0.
        val_loss = 0.
        loss_dict_train = {
            'latent_mse': 0.,
            'image_l1': 0.,
            'loss': 0.,
            'landmarks': 0.
        }

        # while i_iter < n_iters:
        while self.global_step < n_iters:
            for batch in data_loaders['train']:
                # Update learning rate
                # t = i_iter / n_iters
                t = self.global_step / n_iters
                self.update_lr(t)

                # Unpack batch
                audio, input_latent, aux_input, target_latent, target_img = self.unpack_data(
                    batch)

                # Encode
                pred = self.forward(audio, input_latent, aux_input)

                # Compute perceptual loss
                losses = self.get_loss(pred, target_latent, target_img, validate=False)
                loss = losses['loss']

                # Optimize
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                for key, value in losses.items():
                    loss_dict_train[key] += value.item()
                pbar_avg_train_loss += loss.item()

                self.global_step += 1
                i_iter += 1
                pbar.update()

                if self.about_time(self.args.log_val_every):
                    loss_dict_val = self.validate(data_loaders)
                    val_loss = loss_dict_val['loss']

                if self.about_time(self.args.update_pbar_every):
                    pbar_avg_train_loss /= self.args.update_pbar_every
                    pbar.set_description('step [{gs}/{ni}] - '
                                         't-loss {tl:.3f} - '
                                         'v-loss {vl:.3f} - '
                                         'lr {lr:.6f}'.format(
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
                    if self.about_time(self.args.log_train_every):
                        for key in loss_dict_train.keys():
                            loss_dict_train[key] /= max(1, float(self.args.log_train_every))
                            self.train_writer.add_scalar(
                                key, loss_dict_train[key], self.global_step)
                            loss_dict_train[key] = 0.

                    if self.about_time(self.args.log_val_every):
                        for key in loss_dict_val.keys():
                            self.val_writer.add_scalar(
                                key, loss_dict_val[key], self.global_step)

                    if self.about_time(self.args.save_every):
                        self.save()

                    if self.about_time(self.args.eval_every):
                        self.eval(data_loaders['train'], f'train_gen_{self.global_step}.png')
                        self.eval(data_loaders['val'], f'val_gen_{self.global_step}.png')

                # Break if n_iters is reached and still in epoch
                # if i_iter == n_iters:
                if self.global_step == n_iters:
                    break

        self.save()
        print('Done.')

    def validate(self, data_loaders):
        loss_dict = {
            'loss': 0.,
            'latent_mse': 0.,
            'image_l1': 0.,
            'landmarks': 0.
        }
        for batch in data_loaders['val']:
            # Unpack batch
            audio, input_latent, aux_input, target_latent, target_img = self.unpack_data(
                batch)

            with torch.no_grad():
                # Forward
                pred = self.forward(audio, input_latent, aux_input)
                loss = self.get_loss(pred, target_latent, target_img, validate=True)
                for key, value in loss.items():
                    loss_dict[key] += value.item()

        for key in loss_dict.keys():
            loss_dict[key] /= float(len(data_loaders['val']))
        return loss_dict

    def eval(self, data_loader, sample_name):
        # Unpack batch
        batch = next(iter(data_loader))
        audio, input_latent, aux_input, target_latent, target_img, _ = self.unpack_data(
            batch)

        n_display = min(4, self.args.batch_size)
        audio = audio[:n_display]
        target_latent = target_latent[:n_display]
        target_img = target_img[:n_display]
        input_latent = input_latent[:n_display]
        aux_input = aux_input[:n_display]

        with torch.no_grad():
            # Forward
            pred = self.forward(audio, input_latent, aux_input)
            input_img, _ = self.g([input_latent], input_is_latent=True, noise=self.g.noises)
            input_img = utils.downsample_256(input_img)

            pred, _ = self.g(
                [pred], input_is_latent=True, noise=self.g.noises)
            pred = utils.downsample_256(pred)
            target_img, _ = self.g(
                [target_latent], input_is_latent=True, noise=self.g.noises)
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

    def test_model(self, paths, n_test, frames, mode=""):
        counter = 0
        for path in paths:
            split = path[0].split('/')
            sentence = '/'.join(split[:-1]) + '/'
            if split[-2].startswith('TV'):
                continue
            if self.args.random_inp_latent:
                latent = random.choice(glob(sentence + '*.latent.pt'))
            else:
                latent = sentence + 'mean.latent.pt'
            audio_file = '/'.join(split[:-3] + ['AudioMP3'] + [split[-2]]) + '.mp3'
            self.test_video(latent, sentence, audio_file, frames, mode=mode)
            counter += 1
            if counter == n_test:
                break

    def test_video(self, test_latent_path, test_sentence_path, audio_file_path, frames=-1, mode=""):
        print(f"Testing {test_sentence_path}\n{audio_file_path}\n")
        self.audio_encoder.eval()
        if test_sentence_path[-1] != '/':
            test_sentence_path += '/'
        test_latent = torch.load(test_latent_path).unsqueeze(0).to(self.device)
        aux_input = test_latent[:, 4:8]

        sentence_name = test_sentence_path.split('/')[-2]

        # Load audio features
        audio_type = 'deepspeech' if self.args.audio_type == 'deepspeech-synced' else self.args.audio_type
        audio_paths = sorted(glob(test_sentence_path + f'*.{audio_type}.npy'))[:frames]
        audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32) for p in audio_paths]).to(self.device)
        # Pad audio features
        pad = self.args.T // 2
        audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
        audios = audios.unfold(0, self.args.T, 1).permute(0, 3, 1, 2)

        target_latent_paths = sorted(glob(test_sentence_path + '*.latent.pt'))[:frames]
        target_latents = torch.stack([torch.load(p) for p in target_latent_paths]).to(self.device)

        pbar = tqdm(total=len(target_latents))
        video = []

        # Generate
        for i, (audio, target_latent) in enumerate(zip(audios, target_latents)):
            audio = audio.unsqueeze(0)
            target_latent = target_latent.unsqueeze(0)
            with torch.no_grad():
                input_latent = test_latent.clone()
                latent = self.forward(audio, input_latent, aux_input)
                # Generate images
                pred = self.g([latent], input_is_latent=True, noise=self.g.noises)[0]
                # target_img = self.g([target_latent], input_is_latent=True, noise=self.g.noises)[0]
                # Downsample
                pred = utils.downsample_256(pred)
                # target_img = utils.downsample_256(target_img)
            pbar.update()
            # Normalize
            pred = make_grid(pred.cpu(), normalize=True, range=(-1, 1))
            # target_img = make_grid(target_img.cpu(), normalize=True, range=(-1, 1))
            # diff = (target_img - pred) * 5

            # save_tensor = torch.stack((pred, target_img, diff), dim=0)
            # video.append(make_grid(save_tensor))
            video.append(make_grid(pred))

        # Save frames as video
        video = torch.stack(video, dim=0)
        video_name = f"{self.args.save_dir}results/{mode}{sentence_name}"
        os.makedirs(f"{self.args.save_dir}results", exist_ok=True)
        utils.write_video(f'{video_name}.mp4', video, fps=25)

        # Add audio
        p = Popen(['ffmpeg', '-y', '-i', f'{video_name}.mp4', '-i', audio_file_path, '-codec', 'copy', '-shortest', f'{video_name}.mov'],
                  stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        if p.returncode != 0:
            print("Adding audio from %s to video %s failed with error\n%d %s %s" % (
                  audio_file_path, f'{video_name}.mp4', p.returncode, output, error))
        os.system(f"rm {video_name}.mp4")

        self.audio_encoder.train()


def load_data(args):
    # Load data
    train_paths = datasets.get_video_paths_by_file(
        args.data_path, args.train_paths_file, args.max_frames_per_vid)
    val_paths = datasets.get_video_paths_by_file(
        args.data_path, args.val_paths_file, args.max_frames_per_vid)
    test_paths = datasets.get_video_paths_by_file(
        args.data_path, args.test_paths_file, args.max_frames_per_vid)

    if args.overfit:
        train_paths = [train_paths[0]]
        val_paths = train_paths
        print(f"OVERFITTING ON {train_paths[0][0]}")

    print("Sample training videos")
    for i in range(5):
        print(train_paths[i][0])
    print(f"Sample validation videos")
    for i in range(5):
        print(val_paths[i][0])

    train_ds = datasets.AudioVisualDataset(
        paths=train_paths,
        audio_type=args.audio_type,
        load_img=True,
        load_latent=True,
        random_inp_latent=args.random_inp_latent,
        T=args.T,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    val_ds = datasets.AudioVisualDataset(
        paths=val_paths,
        audio_type=args.audio_type,
        load_img=True,
        load_latent=True,
        random_inp_latent=args.random_inp_latent,
        T=args.T,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=256,
    )
    train_sampler = datasets.RandomAudioSampler(
        train_paths, args.T, args.batch_size, 10000, weighted=True, static_random=args.static_random_inp_latent)
    val_sampler = datasets.RandomAudioSampler(
        val_paths, args.T, args.batch_size, 50, weighted=True, static_random=args.static_random_inp_latent)

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
    return data_loaders, train_paths, val_paths, test_paths


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # GPU
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--overfit', action='store_true')

    # Hparams
    parser.add_argument('--batch_size', type=int, default=4)  # 4
    parser.add_argument('--lr', type=int, default=0.0001)  # 0.0001
    parser.add_argument('--T', type=int, default=8)  # 8
    parser.add_argument('--max_frames_per_vid', type=int, default=-1)  # -1
    parser.add_argument('--audio_type', type=str, default='deepspeech-synced')  # 'deepspeech', 'deepspeech-synced'
    parser.add_argument('--random_inp_latent', type=bool, default=False)
    parser.add_argument('--static_random_inp_latent', type=bool, default=False)
    parser.add_argument('--image_loss_type', type=str, default='lpips')  # 'lpips' or 'l1'

    parser.add_argument('--test_multiplier', type=float, default=2.0)  # During test time, direction is multiplied with
    parser.add_argument('--test_truncation', type=float, default=.8)  # After multiplication, truncate to mean latent

    # Loss weights
    parser.add_argument('--latent_loss_weight', type=float, default=1.)  # 1.
    parser.add_argument('--photometric_loss_weight', type=float, default=250.)  # 250.

    # Logging args
    parser.add_argument('--n_iters', type=int, default=150000)
    parser.add_argument('--update_pbar_every', type=int, default=100)  # 100
    parser.add_argument('--log_train_every', type=int, default=200)  # 200
    parser.add_argument('--log_val_every', type=int, default=200)  # 200
    parser.add_argument('--save_every', type=int, default=10000)  # 10000
    parser.add_argument('--eval_every', type=int, default=10000)  # 10000
    parser.add_argument('--save_dir', type=str, default='saves/audio_encoder/')

    # Path args
    parser.add_argument('--data_path', type=str,
                        default=f'{DATAROOT}AudioVisualDataset/Aligned256/')
    parser.add_argument('--train_paths_file', type=str,
                        default=f'{DATAROOT}AudioVisualDataset/split_files/train_videos.txt')
    parser.add_argument('--val_paths_file', type=str,
                        default=f'{DATAROOT}AudioVisualDataset/split_files/val_videos.txt')
    parser.add_argument('--test_paths_file', type=str,
                        default=f'{DATAROOT}AudioVisualDataset/split_files/test_videos.txt')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    if args.cont or args.test:
        assert args.model_path is not None

    # Correct path
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    args.save_dir += timestamp

    if args.cont or args.test:
        args.save_dir = '/'.join(args.model_path.split('/')[:-2]) + '/'

    if args.debug:
        print("DEBUG MODE. NO LOGGING")
    elif args.test:
        print("Testing")

    # Select device
    args.device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.device)

    # Load data
    data_loaders, train_paths, val_paths, test_paths = load_data(args)

    # Init solver
    solver = Solver(args)

    # Train
    if args.test:
        solver.test_model(test_paths, n_test=-1, frames=100, mode='test_')

        # GRID videos
        grid_paths = []
        with open(RAIDROOT + 'Datasets/GRID/grid_videos.txt', 'r') as f:
            line = f.readline()
            while line:
                video = line.replace('\n', '')
                video_root = RAIDROOT + f'Datasets/GRID/Aligned256/{video}/'
                grid_paths.append(sorted(glob(video_root + '*.png')))
                line = f.readline()
        random.shuffle(grid_paths)
        solver.test_model(grid_paths, n_test=-1, frames=-1, mode='')

        # CREMA-D videos
        grid_paths = []
        with open(RAIDROOT + 'Datasets/CREMA-D/crema-d_videos.txt', 'r') as f:
            line = f.readline()
            while line:
                video = line.replace('\n', '')
                video_root = RAIDROOT + f'Datasets/CREMA-D/Aligned256/{video}/'
                grid_paths.append(sorted(glob(video_root + '*.png')))
                line = f.readline()
        random.shuffle(grid_paths)
        solver.test_model(grid_paths, n_test=-1, frames=-1, mode='')
    else:
        solver.train(data_loaders, args.n_iters)
        print("Finished training.")
