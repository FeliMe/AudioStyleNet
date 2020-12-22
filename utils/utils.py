"""
File for general usefull functions which are not specific to a certain module
"""

import cv2
import face_alignment
import numpy as np
import os
import torch

from argparse import Namespace
from imageio import mimwrite
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Dict, Any

HOME = os.path.expanduser('~')
RAIDROOT = os.environ.get('RAIDROOT')


def torch2np_img(img):
    """
    Converts a pytorch image into a cv2 RGB image

    :param img: torch.tensor, range (-1, 1), dtype torch.float32, shape (C, H, W)
    :returns img: np.array, range(0, 255), dtype np.uint8, shape (H, W, C)
    """
    return (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)


def np2torch_img(img):
    """
    Converts a cv2 RGB image into a pytorch image

    :param img: np.array, range(0, 255), dtype np.uint8, shape (H, W, C)
    :returns img: torch.tensor, range (-1, 1), dtype torch.float32, shape (C, H, W)
    """
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


class Downsample(object):
    """ Custom transform: Downsamples image in StyleGAN manner """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        c, h, w = sample.shape
        if h > self.size:
            factor = h // self.size
            sample = sample.reshape(
                c, h // factor, factor, w // factor, factor)
            sample = sample.mean([2, 4])
        return sample


def downsample_256(img):
    b, c, h, w = img.shape
    if h > 256:
        factor = h // 256
        img = img.reshape(b, c, h // factor, factor, w // factor, factor)
        img = img.mean([3, 5])
    return img


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VideoAligner:
    def __init__(self, device):
        # Init face tracking
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=device)

        # Init alignment variables
        self.avg_rotation = 0.
        self.qsize = None
        self.initial_rot = None
        self.prev_qsize = None

    def reset(self):
        self.avg_rotation = 0.
        self.qsize = None
        self.initial_rot = None
        self.prev_qsize = None

    def get_landmarks(self, frame):
        preds = self.fa.get_landmarks(frame)
        if preds is None:
            return None
        return preds[0]

    @staticmethod
    def load_video(videofile):
        frames = []
        cap = cv2.VideoCapture(videofile)
        i_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            i_frame += 1
        assert len(frames) > 0, f"Failed to load {videofile}"
        return np.array(frames)

    @staticmethod
    def get_rotation(v):
        return np.arctan2(v[1], v[0])

    @staticmethod
    def Rotate2D(pts, c, ang=np.pi / 4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return np.dot(pts - c, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + c

    def align_single_image(self, image, save_path, output_size=256):
        if type(image) is str:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = self.get_landmarks(image)

        if landmarks is None:
            print(f"Error, no face found")
            raise RuntimeError

        image = self.align_image(
            image,
            landmarks,
            output_size=256,
            transform_size=1024
        )

        image.save(save_path)

    def align_image(self,
                    frame,
                    landmarks,
                    output_size=1024,
                    transform_size=4096,
                    enable_padding=True):
        """
        Source: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        """

        # Parse landmarks.
        # pylint: disable=unused-variable
        lm = np.array(landmarks)
        # lm_chin = lm[0: 17]  # left-right
        # lm_eyebrow_left = lm[17: 22]  # left-right
        # lm_eyebrow_right = lm[22: 27]  # left-right
        # lm_nose = lm[27: 31]  # top-down
        # lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        mouth_avg_top = np.array([mouth_avg[0], lm_mouth_inner[:, 1].min()])
        # eye_to_mouth = mouth_avg - eye_avg
        eye_to_mouth = mouth_avg_top - eye_avg

        # Choose oriented crop rectangle.
        xq = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        xq /= np.hypot(*xq)
        xq *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        yq = np.flipud(xq) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - xq - yq, c - xq + yq,
                         c + xq + yq, c + xq - yq])

        qsize_raw = np.linalg.norm(quad[1] - quad[0])
        if self.qsize is None:
            self.prev_qsize = qsize_raw

        # Reset if cut in Video (qsize makes a jump)
        if max(qsize_raw / self.prev_qsize, self.prev_qsize / qsize_raw) > 1.3:
            self.prev_qsize = qsize_raw
            self.avg_rotation = 0.
            self.initial_rot = None

        self.qsize = 0.01 * qsize_raw + 0.99 * self.prev_qsize
        self.prev_qsize = self.qsize

        factor = ((self.qsize / qsize_raw) - 1) * 0.5
        # Correct qsize horizontal
        quad[0] -= (quad[3] - quad[0]) * factor
        quad[3] += (quad[3] - quad[0]) * factor
        quad[1] -= (quad[2] - quad[1]) * factor
        quad[2] += (quad[2] - quad[1]) * factor
        # Correct qsize vertical
        quad[0] -= (quad[1] - quad[0]) * factor
        quad[1] += (quad[1] - quad[0]) * factor
        quad[3] -= (quad[2] - quad[3]) * factor
        quad[2] += (quad[2] - quad[3]) * factor

        rotation = self.get_rotation(quad[3] - quad[0])
        if self.initial_rot is None:
            self.initial_rot = rotation

        self.avg_rotation = 0.7 * self.avg_rotation + \
            0.3 * (self.initial_rot - rotation)
        quad = self.Rotate2D(quad, c, self.initial_rot -
                             rotation - self.avg_rotation)

        # Convert image to PIL
        img = Image.fromarray(frame)

        # Shrink.
        shrink = int(np.floor(self.qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(
                float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            self.qsize /= shrink

        # Crop.
        border = max(int(np.rint(self.qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
            np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(
            np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] -
                                                                       img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(self.qsize * 0.3)))
            img = np.pad(np.float32(
                img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(
                w - 1 - x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = self.qsize * 0.02
            img += (gaussian_filter(img, [blur, blur, 0]) -
                    img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * \
                np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(
                np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size),
                            Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.ANTIALIAS)

        return img

    def align_video(self, path_to_vid, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        self.reset()
        i_frame = 0

        video = self.load_video(path_to_vid)

        for frame in video:
            i_frame += 1
            # pbar.update()
            name = str(i_frame).zfill(5) + '.png'
            save_path = os.path.join(save_dir, name)
            if os.path.exists(save_path):
                continue

            landmarks = self.get_landmarks(frame)

            if landmarks is None:
                print(f"No face found in {i_frame}, skipping")
                continue

            frame = self.align_image(
                frame,
                landmarks,
                output_size=256,
                transform_size=1024
            )

            # Visualize
            # print(save_path)
            # frame.show()
            # 1 / 0

            # Save
            frame.save(save_path)


class HparamWriter(SummaryWriter):
    """
    Tensorboard SummaryWriter with support to log argparse parameters.
    For more information about the classic SummaryWriter functionality, please
    refer to:
    https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter
    """

    def __init__(self, logdir):
        super(HparamWriter, self).__init__(logdir)

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @staticmethod
    def _flatten_dict(params: Dict[str, Any], delimiter: str = '/') -> Dict[str, Any]:
        """Flatten hierarchical dict e.g. {'a': {'b': 'c'}} -> {'a/b': 'c'}.
        Args:
            params: Dictionary contains hparams
            delimiter: Delimiter to express the hierarchy. Defaults to '/'.
        Returns:
            Flatten dict.
        Examples:
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, dict):
                for key, value in input_dict.items():
                    if isinstance(value, (dict, Namespace)):
                        value = vars(value) if isinstance(
                            value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns params with non-primitvies converted to strings for logging
        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(LightningLoggerBase._sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
         'float': 0.3,
         'int': 1,
         'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
         'list': '[1, 2, 3]',
         'namespace': 'Namespace(foo=3)',
         'string': 'abc'}
        """
        return {k: v if type(v) in [bool, int, float, str, torch.Tensor] else str(v) for k, v in params.items()}

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Log hyperparameters in form of a Dict or Namespace object to tensorboard

        :param params: Dict or Namespace object. Contains training parameters
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        sanitized_params = self._sanitize_params(params)

        from torch.utils.tensorboard.summary import hparams
        exp, ssi, sei = hparams(sanitized_params, {})
        writer = self._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)


def write_video(path, video, fps):
    """
    Save a sequence of torch tensors of np arrays as video to path

    :param path (str): save path
    :param video (torch.tensor of np.array): frames in correct order
    :param fps: Target fps of video
    """
    if torch.is_tensor(video):
        video = np.transpose(video.data.numpy() * 255.,
                             [0, 2, 3, 1]).astype(np.uint8)
    mimwrite(path, video, fps=fps)
