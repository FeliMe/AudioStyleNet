import argparse
import contextlib
import numpy as np
import os
import shutil
import tempfile
import torch
import torch.nn.functional as F

from glob import glob
from my_models import models
from my_models.style_gan_2 import PretrainedGenerator1024
from subprocess import Popen, PIPE
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import utils


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)

    with cd(dirpath, cleanup):
        yield dirpath


class Emotion_Aware_Facial_Animation:
    def __init__(self,
                 model_path,
                 device,
                 model_type='net3',
                 audio_type='deepspeech',
                 T=8,
                 n_latent_vec=4,
                 normalize_audio=False):

        self.device = device
        torch.cuda.set_device(device)
        self.n_latent_vec = n_latent_vec
        self.normalize_audio = normalize_audio
        self.T = T
        self.audio_type = audio_type

        # Init Generator
        self.g = PretrainedGenerator1024().eval().to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

        # Define audio encoder
        if model_type == 'net2':
            self.audio_encoder = models.AudioExpressionNet2(
                T, n_latent_vec).to(self.device).eval()
        elif model_type == 'net3':
            self.audio_encoder = models.AudioExpressionNet3(
                T, n_latent_vec).to(self.device).eval()
        elif model_type == 'net4':
            self.audio_encoder = models.AudioExpressionNet4(
                T, n_latent_vec).to(self.device).eval()
        elif model_type == 'net5':
            self.audio_encoder = models.AudioExpressionNet5(
                T, n_latent_vec).to(self.device).eval()
        else:
            raise NotImplementedError

        if audio_type == 'lpc':
            self.audio_encoder = models.AudioExpressionNet4(
                T, n_latent_vec).to(self.device).eval()
        elif audio_type == 'mfcc':
            self.audio_encoder = models.AudioExpressionSyncNet(
                T, n_latent_vec).to(self.device).eval()

        # Load weights
        self.load(model_path)

    def load(self, path):
        print(f"Loading audio_encoder weights from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        if type(checkpoint) == dict:
            self.audio_encoder.load_state_dict(checkpoint['model'])
        else:
            self.audio_encoder.load_state_dict(checkpoint)

    def forward(self,
                audio,
                input_latent,
                aux_input,
                direction=None,
                audio_multiplier=2.,
                audio_truncation=.8,
                direction_multiplier=1.):
        # Normalize audio features
        if self.normalize_audio:
            audio = (audio - audio.mean()) / audio.std()

        # Predict offset
        if self.n_latent_vec == 4:
            latent_offset = self.audio_encoder(audio, aux_input)
            prediction = input_latent.clone()

            # Adapt strength of direction
            latent_offset *= audio_multiplier
            prediction[:, 4:8] += latent_offset

            # Truncation trick
            prediction[:, 4:8] = self.g.latent_avg + audio_truncation * \
                (prediction[:, 4:8] - self.g.latent_avg)

            # Add another direction to the prediction
            if direction is not None:
                prediction[:, :8] += (direction * direction_multiplier)

        elif self.n_latent_vec == 8:
            latent_offset = self.audio_encoder(audio, aux_input)
            prediction = input_latent.clone()

            # Adapt strength of direction
            latent_offset *= audio_multiplier
            prediction[:, 4:8] += latent_offset

            # Truncation trick
            prediction[:, 4:8] = self.g.latent_avg + audio_truncation * \
                (prediction[:, 4:8] - self.g.latent_avg)

            # Add another direction to the prediction
            if direction is not None:
                prediction[:, :8] += (direction * direction_multiplier)

        else:
            raise NotImplementedError

        return prediction

    def __call__(self,
                 test_latent,
                 test_sentence_path,
                 direction=None,
                 use_landmark_input=False,
                 audio_multiplier=2.0,
                 audio_truncation=0.8,
                 direction_multiplier=1.0,
                 max_sec=None):
        # Load test latent
        if type(test_latent) is str:
            test_latent = torch.load(test_latent).unsqueeze(0).to(self.device)
        else:
            test_latent = test_latent.unsqueeze(0).to(self.device)

        # Auxiliary input
        if use_landmark_input:
            aux_input = torch.cat((torch.load(test_latent.split(
                '.')[0] + '.landmarks.pt').float().view(-1).to(self.device), test_latent[:, 4:8].view(-1))).unsqueeze(0)
        else:
            if self.n_latent_vec == 4:
                aux_input = test_latent[:, 4:8]
            elif self.n_latent_vec == 8:
                aux_input = test_latent[:, :8]
            else:
                raise NotImplementedError

        # Load audio features
        audio_paths = sorted(glob(test_sentence_path + f'*.{self.audio_type}.npy'))
        audios = torch.stack([torch.tensor(np.load(p), dtype=torch.float32)
                              for p in audio_paths]).to(self.device)

        if max_sec is not None:
            max_frames = 25 * max_sec
            audios = audios[:max_frames]

        # Pad audio features
        pad = self.T // 2
        audios = F.pad(audios, (0, 0, 0, 0, pad, pad - 1), 'constant', 0.)
        audios = audios.unfold(0, self.T, 1).permute(0, 3, 1, 2)

        # Load direction if provided
        if direction is not None:
            # Load test latent
            if type(direction) is str:
                ext = direction.split('.')[-1]
                if ext == 'npy':
                    direction = torch.tensor(
                        np.load(direction), dtype=torch.float32).unsqueeze(0).to(self.device)
                elif ext == 'pt':
                    direction = torch.load(direction).unsqueeze(0).to(self.device)
                else:
                    raise RuntimeError
            else:
                direction = direction.unsqueeze(0).to(self.device)

        # pbar = tqdm(total=len(audios))
        video = []

        # Generate
        for i, audio in enumerate(audios):
            audio = audio.unsqueeze(0)
            with torch.no_grad():
                input_latent = test_latent.clone()
                latent = self.forward(audio, input_latent, aux_input, direction,
                                      audio_multiplier=audio_multiplier,
                                      audio_truncation=audio_truncation,
                                      direction_multiplier=direction_multiplier)

                # Generate images
                pred = self.g([latent], input_is_latent=True, noise=self.g.noises)[0]

                # Downsample
                pred = utils.downsample_256(pred)
            # pbar.update()

            # Normalize
            pred = make_grid(pred.cpu(), normalize=True, range=(-1, 1))
            video.append(pred)

        return torch.stack(video)

    def save_video(self, video, audiofile, f):
        print(f"Saving to {f}")
        with tempdir() as tmp_path:
            # Save frames as video
            utils.write_video(f'{tmp_path}/tmp.avi', video, fps=25)

            # Add audio
            p = Popen(['ffmpeg', '-y', '-i', f'{tmp_path}/tmp.avi', '-i', audiofile, '-codec', 'copy', '-shortest', f],
                      stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            if p.returncode != 0:
                print("Adding audio from %s to video %s failed with error\n%d %s %s" % (
                    audiofile, f'{tmp_path}/tmp.avi', p.returncode, output, error))


if __name__ == '__main__':

    # Model args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saves/audio_encoder/old_split/lpips_latentmse_all_net3_100k/models/model100000.pt')
    parser.add_argument('--model_type', type=str, default='net3')
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--n_latent_vec', type=int, default=4)
    parser.add_argument('--normalize_audio', type=bool, default=False)
    model_args = parser.parse_args()

    # Sentence args
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_latent', type=str, required=True)
    parser.add_argument('--test_sentence', type=str, required=True)
    parser.add_argument('--audiofile', type=str, required=True)
    sentence_args = parser.parse_args()

    model = Emotion_Aware_Facial_Animation(
        model_path=model_args.model_path,
        model_type=model_args.model_type,
        audio_type=model_args.audio_type,
        T=model_args.T,
        n_latent_vec=model_args.n_latent_vec,
        normalize_audio=model_args.normalize_audio
    )

    model.predict(sentence_args.test_latent, sentence_args.test_sentence, sentence_args.audiofile)
