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


class AudioStyleNet:
    def __init__(self,
                 model_path,
                 device,
                 audio_type='deepspeech',
                 T=8):

        self.device = device
        torch.cuda.set_device(device)
        self.T = T
        self.audio_type = audio_type

        # Init Generator
        self.g = PretrainedGenerator1024().eval().to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

        # Define audio encoder
        self.audio_encoder = models.AudioExpressionNet3(
            T).to(self.device).eval()

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

        # Predict offset
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
            prediction[:, :8] += (direction * direction_multiplier)  # TODO: +-

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

        if test_latent.shape[1] == 1:
            test_latent = test_latent.repeat(1, 18, 1)

        # Visualize
        # img = self.g([test_latent], input_is_latent=True, noise=self.g.noises)[0]
        # img = utils.downsample_256(img)
        # img = make_grid(img.cpu(), normalize=True, range=(-1, 1))
        # from torchvision import transforms
        # transforms.ToPILImage('RGB')(img).show()
        # 1 / 0

        # Auxiliary input
        aux_input = test_latent[:, 4:8]

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

            # Normalize
            pred = make_grid(pred.cpu(), normalize=True, range=(-1, 1))
            video.append(pred)

        return torch.stack(video)

    def save_video(self, video, audiofile, f):
        print(f"Saving to {f}")
        if not os.path.isabs(audiofile):
            audiofile = os.path.join(os.getcwd(), audiofile)
        if not os.path.isabs(f):
            f = os.path.join(os.getcwd(), f)
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
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--T', type=int, default=8)
    model_args = parser.parse_args()

    # Sentence args
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_latent', type=str, required=True)
    parser.add_argument('--test_sentence', type=str, required=True)
    parser.add_argument('--audiofile', type=str, required=True)
    sentence_args = parser.parse_args()

    model = Emotion_Aware_Facial_Animation(
        model_path=model_args.model_path,
        audio_type=model_args.audio_type,
        T=model_args.T
    )

    model.predict(sentence_args.test_latent, sentence_args.test_sentence, sentence_args.audiofile)
