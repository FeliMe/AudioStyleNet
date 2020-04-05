"""
File to specify dataloaders for different datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import torch
import torch.nn.functional as F

from glob import glob
from my_models.style_gan_2 import Generator
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from torchvision.utils import make_grid
from utils.utils import downsample_256


MAPPING = {
    'neutral': '01',
    'calm': '02',
    'happy': '03',
    'sad': '04',
    'angry': '05',
    'fearful': '06',
    'disgust': '07',
    'surprised': '08'
}


class ImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256):
        super().__init__()
        self.normalize = normalize
        self.mean = mean
        self.std = std

        print(f"Searching data in {root_path}")
        self.paths = sorted(glob(root_path + '*/*.png'))
        assert len(self.paths) > 0, "ImageDataset is empty"

        random.shuffle(self.paths)

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.t(Image.open(self.paths[index]))
        return {'img': img}


class RAVDESSDataset(Dataset):
    """
    Dataset class for loading RAVDESS sentences

    Output shapes:
        [batch_size, 1 or 3, height, width]

    Arguments:
        root_path (str): Path to data files
        normalize (bool): Normalize data before outputting
        mean (list): Dataset mean values
        std (list): Dataset standard deviations
        image_size (int or tuple): Size of input images
        label_one_hot (bool): Choose if emotion is scalar or one_hot vector
        emotions (list of str): List of emotions to be considered
        actors (list of int): List of actors to be considered
    """
    def __init__(self,
                 root_path,
                 use_mask=False,
                 mask_path=None,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=64,
                 num_classes=8,
                 label_one_hot=False,
                 emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                           'fearful', 'disgust', 'surprised'],
                 actors=[i + 1 for i in range(24)]):

        self.use_mask = use_mask
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.num_classes = max(num_classes, len(emotions))
        self.label_one_hot = label_one_hot

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        sentences = [str(p) for p in list(root_dir.glob('*/*'))
                     if str(p).split('/')[-1] != '.DS_Store']

        # Check if not empty
        if len(sentences) == 0:
            raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

        # Filter included actors
        sentences = filter_actor(sentences, actors)
        print("Actors included in data: {}".format(actors))

        # Filter senteces by emotions
        self.emotions = [MAPPING[e] for e in emotions]
        sentences = filter_emotion(self.emotions, sentences)
        print("Emotions included in data: {}".format(
            [list(MAPPING.keys())[list(MAPPING.values()).index(e)] for e in self.emotions]))

        # Get all frames from selected sentences
        sentences = [sorted([str(p) for p in list(pathlib.Path(s).glob('*')) if str(p).split('/')[-1] != '.DS_Store'])
                     for s in sentences]

        # Count length of all sentences
        len_sentences = [len(s) for s in sentences]

        # Count number of frames for every emotion
        # tmp = [f.split('/')[-2].split('-')[2] for f in frames]
        # for emo in emotions:
        #     print("# frames for '{}': {}".format(
        #         emo, len(list(filter(lambda t: t == self.mapping[emo], tmp)))))

        # Shuffle frames
        random.shuffle(sentences)
        # random.shuffle(frames)

        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]

        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

        self.sentences = sentences
        self.len_sentences = len_sentences

        if self.use_mask:
            self.transform_mask = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
            self.mask_path = mask_path

        self.load_fn = Image.open

    def _get_sample(self, sentence, idx):
        path = os.path.join(sentence, str(idx).zfill(3) + '.png')
        x = self.load_fn(path, self.t)
        return x

    def _get_mask(self, sentence, idx):
        path = os.path.join(self.mask_path, *sentence.split('/')
                            [-2:], str(idx).zfill(3) + '.png')
        mask = self.transform_mask(Image.open(path))
        return mask

    def _get_random_idx(self, sentence):
        len_sentence = len(list(pathlib.Path(sentence).glob('*')))
        rand_idx = torch.randint(1, len_sentence + 1, (1,)).item()
        return rand_idx

    def __len__(self):
        return len(self.sentences)
        # return len(self.frames)

    def __getitem__(self, index):

        # Select a sentence
        sentence = self.sentences[index]

        # Get sample
        rand_idx = random.randint(0, self.len_sentences[index] - 1)
        x = self.t(Image.open(sentence[rand_idx]))

        # Load mask
        if self.use_mask:
            mask = self._get_mask(sentence, rand_idx)
        else:
            mask = 0

        # Get emotion
        emotion = int(sentence.split('/')[-1].split('-')[2]) - 1
        if self.label_one_hot:
            emotion = int_to_one_hot(emotion)

        return {'x': x, 'y': emotion, 'mask': mask}


class RAVDESSDSPix2Pix(RAVDESSDataset):
    def __init__(self,
                 root_path,
                 target_root_path,
                 data_format='image',
                 use_same_sentence=True,
                 normalize=True,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 seed=999,
                 image_size=64,
                 label_one_hot=False,
                 emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                           'fearful', 'disgust', 'surprised'],
                 actors=[i + 1 for i in range(24)]):
        super(RAVDESSDSPix2Pix, self).__init__(root_path, data_format,
                                               normalize, mean, std, seed,
                                               image_size, label_one_hot,
                                               emotions, actors)

        self.target_root_path = target_root_path
        self.use_same_sentence = use_same_sentence
        self.show_fn = show_pix2pix

        if normalize:
            print("Mean: {}, std: {}".format(self.mean, self.std))
        else:
            print("No normalization")

    def __getitem__(self, index):
        """
        Gets a pair of sequences (input b and target a). Source of the sequences
        is defined by root_path for input and target_root_path for target
        """

        # Input frame
        input_frame = self.frames[index]
        a = self.t(Image.open(input_frame))

        # Target frame
        if self.use_same_sentence:
            target_frame = os.path.join(
                self.target_root_path, *input_frame.split('/')[-3:])
        else:
            # Use frame from same actor
            actor = os.path.join(self.target_root_path,
                                 *input_frame.split('/')[-3:-2])
            # Get all frames from actor
            all_frames = [str(p) for p in list(pathlib.Path(actor).glob('*/*.jpg'))]
            # Target frame must have different emotion
            inp_emotion = input_frame.split('/')[-2].split('-')[2]
            emotions = [e for e in self.emotions if e != inp_emotion]
            all_frames = list(filter(lambda s: s.split('/')[-2].split('-')[2]
                                     in emotions, all_frames))
            # Randomly select a frame
            target_frame = random.choice(all_frames)
        b = self.t(Image.open(target_frame))

        emotion = int(target_frame.split('/')[-2].split('-')[2]) - 1
        if self.label_one_hot:
            emotion = int_to_one_hot(emotion)

        return {'A': a, 'B': b, 'y': emotion, 'idx': index}

    def show_sample(self):
        """
        Plot a random sample
        """
        sample = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)


class CELEBADataset(Dataset):
    def __init__(self,
                 root_path,
                 target_path,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 img_size=64,
                 seed=999):
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.show_fn = show_pix2pix
        self.target_path = target_path

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        self.files = [str(p) for p in list(root_dir.glob('*'))
                      if str(p).split('/')[-1] != '.DS_Store']

        # Random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Shuffle sentences
        random.shuffle(self.files)

        trans = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ]
        if self.normalize:
            trans.append(transforms.Normalize(self.mean, self.std))

        self.trans = transforms.Compose(trans)
        self.load_fn = Image.open

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path_a = self.files[item]
        path_b = os.path.join(self.target_path, *path_a.split('/')[-1:])
        a = self.load_fn(path_a, self.trans).unsqueeze(0)
        b = self.load_fn(path_b, self.trans).unsqueeze(0)
        return {'A': a, 'B': b, 'y': 0.}

    def show_sample(self):
        """
        Plot a random sample
        """
        sample = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)


class RAVDESSFlatDataset(Dataset):
    def __init__(self,
                 paths,
                 device,
                 load_landmarks=False,
                 load_latent=False,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 label_one_hot=False):

        self.load_landmarks = load_landmarks
        self.load_latent = load_latent
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.label_one_hot = label_one_hot
        self.device = device

        frames = paths

        # Select load function
        if frames[0].split('.')[-1] in ['jpg', 'png']:
            self.load_fn = Image.open
            if int(np.log2(image_size)) - np.log2(image_size) == 0:
                trans = [transforms.ToTensor(), Downsample(image_size)]
            else:
                trans = [transforms.Resize(image_size), transforms.ToTensor()]
            if self.normalize:
                trans.append(transforms.Normalize(mean=self.mean, std=self.std))
            self.t = transforms.Compose(trans)
        elif frames[0].split('.')[-1] == 'pt':
            self.load_fn = torch.load
            self.t = lambda x: x

        self.frames = frames

        # self.target_happy_scores = self._compute_happy_scores()

    def __len__(self):
        return len(self.frames)

    # def _compute_happy_scores(self):
    #     print("Computing happy scores for target images")
    #     model = FERClassifier().to(self.device)
    #     scores = []
    #     for frame in self.frames:
    #         img = self.t(Image.open(frame)).unsqueeze(0).to(self.device)
    #         score = model(img)[0, 2].cpu().view(1,)
    #         scores.append(score)

    #     # Linearify scores from 0 to 1
    #     lin_scores = [0] * len(scores)
    #     for i, (item, idx) in enumerate(zip(np.linspace(0., 1., num=len(scores)), np.argsort(scores))):
    #         lin_scores[idx] = torch.tensor([item])
    #     scores = lin_scores

    #     return scores

    def __getitem__(self, index):
        # Select frame
        frame = self.frames[index]
        # Get emotion
        emotion = int(frame.split('/')[-2].split('-')[2]) - 1
        if self.label_one_hot:
            emotion = int_to_one_hot(emotion)
        # Load image
        img = self.t(self.load_fn(frame))

        # Load landmarks
        if self.load_landmarks:
            landmarks = torch.load(frame.split('.')[0] + '.landmarks.pt')
            # Normalize landmarks
            landmarks = (landmarks / 127.5) - 1.
        else:
            landmarks = torch.tensor(0.)

        if self.load_latent:
            latent = torch.load(frame.split('.')[0] + '.latent.pt')
        else:
            latent = torch.tensor(0.)

        return {
            'img': img,
            'landmarks': landmarks,
            'latent': latent,
            'y': emotion,
            'index': index,
            'path': frame
        }


class RAVDESSNeutralToXDataset(Dataset):
    def __init__(self,
                 paths,
                 all_paths,
                 device,
                 seed=123,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 emotion_x='angry',
                 lin_scores=False,
                 score_type='fer'):
        super().__init__()
        print("Loading dataset")

        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.device = device
        self.lin_scores = lin_scores
        self.score_type = score_type
        self.emotion_x = emotion_x

        # Filter senteces by emotions
        sentences_neutral_raw = filter_emotion([MAPPING['neutral']], all_paths)
        sentences_x = filter_emotion(MAPPING[self.emotion_x], paths)

        # Get x_neutral pairs
        sentences_neutral = []
        for s in sentences_x:
            ident = s[0].split('/')[-2].split('-')
            ident[2] = MAPPING['neutral']
            ident[3] = '01'
            ident = '-'.join(ident)
            sentences_neutral += list(
                filter(lambda s: s[0].split('/')[-2] == ident, sentences_neutral_raw))

        # Get length of sentences
        len_sentences_neutral = [len(s) for s in sentences_neutral]
        len_sentences_x = [len(s) for s in sentences_x]

        print(f"# sentences neutral: {len(sentences_neutral)}")
        print(f"# sentences x: {len(sentences_x)}")

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

        self.sentences_neutral = sentences_neutral
        self.sentences_x = sentences_x
        self.len_sentences_neutral = len_sentences_neutral
        self.len_sentences_x = len_sentences_x

        self.scores_x = self._load_scores_x()

    def __len__(self):
        return len(self.sentences_x)

    def _load_scores_x(self):
        print("Loading scores for {}".format(self.emotion_x))
        if self.score_type in ['fer', 'ravdess']:
            appendix = f'-logit_{self.score_type}.pt'
        elif self.score_type == 'azure':
            appendix = '-score_azure.pt'
        else:
            raise NotImplementedError
        scores_x = []
        for s in self.sentences_x:
            scores_x.append([])
            for f in s:
                emotion = int(f.split('/')[-2].split('-')[2]) - 1
                score_path = f.split('.')[0] + appendix
                score = torch.load(score_path)
                score = F.softmax(score, dim=0)[emotion]
                scores_x[-1].append(score)
        scores_flat = np.array([it for s in scores_x for it in s])
        print("Min score: {:.4f}; max score: {:.4f}".format(
            scores_flat.min(), scores_flat.max()))

        if self.lin_scores:
            # Linearify scores from 0 to 1
            print("Linearifying scores from 0 to 1")
            lin_scores_flat = np.zeros_like(scores_flat)
            for item, idx in zip(np.linspace(0., 1., num=len(scores_flat)), np.argsort(scores_flat)):
                lin_scores_flat[idx] = item
            lin_scores = []
            counter = 0
            for i in range(len(scores_x)):
                lin_scores.append([])
                for j in range(len(scores_x[i])):
                    lin_scores[-1].append(torch.tensor([lin_scores_flat[counter]]))
                    counter += 1
            scores_x = lin_scores

        return scores_x

    def __getitem__(self, index):
        # Select src and target sentence
        sentence_neutral = self.sentences_neutral[index]
        sentence_x = self.sentences_x[index]

        # Get random index
        rand_idx_neutral = random.randint(
            0, self.len_sentences_neutral[index] - 1)
        rand_idx_x = min(rand_idx_neutral, self.len_sentences_x[index] - 1)

        # Load Images
        neutral = self.t(Image.open(sentence_neutral[rand_idx_neutral]))
        x = self.t(Image.open(sentence_x[rand_idx_x]))

        res = {
            'neutral': neutral,
            'x': x,
            'index': index,
            'rand_x_idx': rand_idx_x,
        }

        res['score_x'] = self.scores_x[index][rand_idx_x]

        return res


class RAVDESSNeutralToAllDataset(Dataset):
    def __init__(self,
                 paths,
                 all_paths,
                 device,
                 score_type='ravdess',
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 emotions=['calm', 'happy', 'sad', 'angry',
                           'fearful', 'disgust', 'surprised']):
        super().__init__()
        print("Loading dataset")

        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.device = device
        self.score_type = score_type

        # Get corresponding indices in scores of emotions
        emotion_inds = [int(MAPPING[e]) - 1 for e in emotions if e != 'neutral']

        # Filter in neutral sentences and rest
        sentences_neutral_raw = filter_emotion([MAPPING['neutral']], all_paths)
        sentences_x_raw = filter_emotion(
            [MAPPING[e] for e in emotions if e != 'neutral'], paths)

        # Get x_neutral pairs
        sentences_x = []
        sentences_neutral = []
        for s in sentences_x_raw:
            sentences_x.append(s)
            ident = s[0].split('/')[-2].split('-')
            ident[2] = MAPPING['neutral']
            ident[3] = '01'
            ident = '-'.join(ident)
            sentences_neutral += list(
                filter(lambda s: s[0].split('/')[-2] == ident, sentences_neutral_raw))

        # Get lengths of sentences
        len_sentences_neutral = [len(s) for s in sentences_neutral]
        len_sentences_x = [len(s) for s in sentences_x]

        # Print number of sentences
        print(f"# sentences neutral {len(sentences_neutral)}")
        print(f"# sentences x {len(sentences_x)}")

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

        self.sentences_neutral = sentences_neutral
        self.sentences_x = sentences_x
        self.len_sentences_neutral = len_sentences_neutral
        self.len_sentences_x = len_sentences_x
        self.emotion_inds = emotion_inds

        self.scores_x = self._load_scores_x()

    def _load_scores_x(self):
        print("Loading scores")
        if self.score_type in ['fer', 'ravdess']:
            appendix = f'-logit_{self.score_type}.pt'
        elif self.score_type == 'azure':
            appendix = '-score_azure.pt'
        else:
            raise NotImplementedError
        scores_x = []
        for s in self.sentences_x:
            scores_x.append([])
            for f in s:
                score_path = f.split('.')[0] + appendix
                score = torch.load(score_path)
                score = F.softmax(score, dim=0)[self.emotion_inds]
                scores_x[-1].append(score)
        return scores_x

    def __len__(self):
        return len(self.sentences_x)

    def __getitem__(self, index):
        # Select src and target sentence
        sentence_neutral = self.sentences_neutral[index]
        sentence_x = self.sentences_x[index]

        # Get random index
        rand_idx_neutral = random.randint(0, self.len_sentences_neutral[index] - 1)
        rand_idx_x = min(rand_idx_neutral, self.len_sentences_x[index] - 1)

        # Load Images
        neutral = self.t(Image.open(sentence_neutral[rand_idx_neutral]))
        x = self.t(Image.open(sentence_x[rand_idx_x]))

        res = {
            'neutral': neutral,
            'x': x,
            'index': index,
            'rand_idx_x': rand_idx_x,
        }

        res['score_x'] = self.scores_x[index][rand_idx_x]

        return res


class RAVDESSEmoDBFlatDataset(Dataset):
    def __init__(self,
                 paths,
                 device,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 seed=123,
                 image_size=256):

        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.device = device

        frames = paths

        # Select load function
        self.load_fn = Image.open
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(self.mean, self.std))
        self.t = transforms.Compose(trans)

        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def get_idx_from_str(self, str_path):
        return [idx for idx, f in enumerate(self.frames) if str_path in f][0]

    def __getitem__(self, index):
        # Check if first frame in a sentence
        if self.frames[index].split('/')[-1].split('.')[0] == '001':
            ind1 = index
            ind2 = index + 1
        else:
            ind1 = index - 1
            ind2 = index
        # Load image
        img1 = self.t(self.load_fn(self.frames[ind1]))
        img2 = self.t(self.load_fn(self.frames[ind2]))

        # Stack images
        imgs = torch.stack((img1, img2), dim=0)
        inds = torch.tensor([ind1, ind2])
        # print(self.frames[ind1], self.frames[ind2], ind1, ind2)

        return {'x': imgs, 'index': inds}


class OMGDataset(Dataset):
    def __init__(self,
                 paths,
                 info,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 flat=False,
                 use_valence=False,
                 binary_valence=False):
        super().__init__()
        self.paths = paths
        self.info = info
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.flat = flat
        self.use_valence = use_valence
        self.binary_valence = binary_valence

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Select utterance
        if self.flat:
            frame = self.paths[index]
        else:
            utterance = self.paths[index]
            frame = random.choice(utterance)

        # Load image
        img = self.t(Image.open(frame))

        # Get emotion label
        vid, utt = frame.split('/')[-3:-1]
        idx = self.info.index[
            (self.info['video'] == vid) & (self.info['utterance'] == utt + '.mp4')][0]
        if self.use_valence:
            valence = self.info.at[idx, 'valence']
            arousal = (self.info.at[idx, 'arousal'] - 0.5) * 2
            if self.binary_valence:
                # valence = round((valence + 1) / 2)
                # arousal = round((arousal + 1) / 2)
                valence = (round((valence + 1) / 2) - 0.5) * 2
                arousal = (round((arousal + 1) / 2) - 0.5) * 2
            emotion = torch.tensor([valence, arousal], dtype=torch.float32)
        else:
            emotion = self.info.at[idx, 'EmotionMaxVote']

        return {'x': img, 'emotion': emotion, 'index': index, 'path': frame}


class AffWild2Dataset(Dataset):
    def __init__(self,
                 paths,
                 annotations,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256):
        super().__init__()
        self.paths = paths
        self.annotations = annotations
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.flat = len(self.paths[0][0]) == 1

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Select utterance
        if self.flat:
            frame = self.paths[index]
        else:
            video = self.paths[index]
            frame = random.choice(video)

        # Load image
        img = self.t(Image.open(frame))

        video = frame.split('/')[-2]
        frame_number = int(frame.split('/')[-1].split('.')[0]) - 1
        emotion = self.annotations[video][frame_number]

        return {'x': img, 'emotion': emotion, 'index': index, 'path': frame}


class TagesschauDataset(Dataset):
    def __init__(self,
                 paths,
                 load_img=True,
                 load_latent=False,
                 load_audio=False,
                 load_mean=False,
                 shuffled=False,
                 flat=False,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256):
        super().__init__()
        self.load_img = load_img
        self.load_latent = load_latent
        self.load_audio = load_audio
        self.load_mean = load_mean
        self.normalize = normalize
        self.mean = mean
        self.std = std

        if flat:
            paths = [item for sublist in paths for item in sublist]

        if shuffled:
            random.shuffle(paths)

        self.paths = paths
        self.flat = len(self.paths[0][0]) == 1

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Select utterance
        if self.flat:
            # if self.overfit:
            #     base_path = '/'.join(self.paths[0].split('/')[:-1]) + '/00001'
            # else:
            #     base_path = self.paths[index]
            base_path = self.paths[index]
        else:
            # if self.overfit:
            #     base_path = self.paths[0] + '00001'
            # else:
            #     video = self.paths[index]
            #     base_path = random.choice(video)
            video = self.paths[index]
            base_path = random.choice(video)

        # Load image
        img = self.t(Image.open(base_path + '.png')) if self.load_img else torch.tensor(0.)
        latent = torch.load(base_path + '.latent.pt') if self.load_latent else torch.tensor(0.)
        audio = torch.tensor(np.load(base_path + '.deepspeech.npy'),
                             dtype=torch.float32) if self.load_audio else torch.tensor(0.)

        # Load input image
        # r_idx = random.randrange(1413)
        # mean_path = '/'.join(base_path.split('/')[:-1]) + f'/{str(r_idx).zfill(5)}.latent.pt'
        mean_path = '/'.join(base_path.split('/')[:-1]) + '/mean.latent.pt'
        mean = torch.load(mean_path) if self.load_mean else torch.tensor(0.)

        return {
            'audio': audio,
            'img': img,
            'target_latent': latent,
            'input_latent': mean,
            'index': index,
            'path': base_path
        }


class TagesschauAudioDataset(Dataset):
    def __init__(self,
                 paths,
                 load_img=True,
                 load_latent=False,
                 T=8,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 len_dataset=None):
        super().__init__()
        self.load_img = load_img
        self.load_latent = load_latent
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.T = T
        self.len_dataset = len_dataset

        self.paths = [item for sublist in paths for item in sublist]

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return self.len_dataset if self.len_dataset else len(self.paths)

    def __getitem__(self, indices):
        audio_inds = indices[:-1]
        input_ind = indices[-1]
        target_ind = indices[self.T // 2]

        video = '/'.join(self.paths[input_ind].split('/')[:-1]) + '/'

        # Load audio
        audio = []
        for i in audio_inds:
            audio.append(torch.tensor(
                np.load(self.paths[i] + '.deepspeech.npy'), dtype=torch.float32))
        audio = torch.stack(audio, dim=0)

        # Load images
        if self.load_img:
            input_img = self.t(Image.open(self.paths[input_ind] + '.png'))
            target_img = self.t(Image.open(self.paths[target_ind] + '.png'))
        else:
            target_img = torch.tensor(0.)
            input_img = torch.tensor(0.)

        # Load latents
        if self.load_latent:
            input_latent = torch.load(video + 'mean.latent.pt')
            # input_latent = torch.load(video + '00001.latent.pt')
            # input_latent = torch.load(self.paths[input_ind] + ".latent.pt")
            target_latent = torch.load(self.paths[target_ind] + ".latent.pt")
        else:
            target_latent = torch.tensor(0.)
            input_latent = torch.tensor(0.)

        return {
            'audio': audio,
            'input_img': input_img,
            'target_img': target_img,
            'input_latent': input_latent,
            'target_latent': target_latent,
            'indices': indices,
            'path': video + str(target_ind).zfill(5)
        }


class StyleGANDataset(IterableDataset):
    def __init__(self, batch_size, downsample=True, device='cuda'):
        super(StyleGANDataset, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.downsample = downsample

        # Init generator
        self.g = Generator(1024, 512, 8, pretrained=True).eval().to(self.device)
        self.g.noises = [n.to(self.device) for n in self.g.noises]
        self.g.latent_avg = self.g.latent_avg.to(self.device)
        for param in self.g.parameters():
            param.requires_grad = False

    def __iter__(self):
        # Sample random z
        z = torch.randn(self.batch_size, 512, device=self.device)

        # Generate image
        with torch.no_grad():
            img, _ = self.g([z], truncation=0.9, truncation_latent=self.g.latent_avg)

        # Resize from 1024 to 256
        if self.downsample:
            img = downsample_256(img)

        yield {'x': img}


class FFHQDataset(Dataset):
    def __init__(self,
                 paths,
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256):
        super().__init__()

        self.paths = paths
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # Transforms
        trans = [transforms.RandomHorizontalFlip()]
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans += [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans += [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.t(Image.open(path))
        return {'img': img, 'path': path, 'index': index}


def show_pix2pix(sample, mean, std, normalize):
    """
    Plots a sample (input sequence and target sequence)
    """
    img_a = sample['A']
    img_b = sample['B']

    # Denormalize
    # if normalize:
    #     transform = utils.denormalize(mean, std)
    #     img_a = transform(img_a)
    #     img_b = transform(img_b)

    # Make image grid
    imgs = torch.stack([img_a, img_b])
    imgs = make_grid(imgs, nrow=img_a.size(0), normalize=True)

    # Plot image
    plt.figure(figsize=(img_a.size(0), 2))
    plt.imshow(np.moveaxis(imgs.numpy(), 0, 2))
    plt.axis('off')
    plt.show()


def get_data_loaders(train_ds, val_ds, batch_size, use_cuda, val_batch_size=None):

    if use_cuda and torch.cuda.is_available():
        print("Pinning memory")
        kwargs = {'pin_memory': True}
    else:
        kwargs = {}

    val_batch_size = val_batch_size if val_batch_size is not None else batch_size

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True,
                              drop_last=True,
                              **kwargs)

    val_loader = DataLoader(val_ds,
                            batch_size=val_batch_size,
                            num_workers=4,
                            shuffle=True,
                            drop_last=True,
                            **kwargs) if val_ds is not None else None

    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }

    dataset_sizes = {
        'train': len(train_ds),
        'val': len(val_ds) if val_ds is not None else None
    }

    return data_loaders, dataset_sizes


def ravdess_get_paths(root_path,
                      flat,
                      shuffled=True,
                      validation_split=0.0,
                      max_frames_per_sentence=None,
                      emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                                'fearful', 'disgust', 'surprised'],
                      actors=[i + 1 for i in range(24)]):
    root_dir = pathlib.Path(root_path)

    # Get paths to all sentences
    sentences = [str(p) for p in list(root_dir.glob('*/*/'))
                 if str(p).split('/')[-1] != '.DS_Store']

    # Check if not empty
    if len(sentences) == 0:
        raise (RuntimeError("Found 0 files in sub-folders of: " + root_path))

    # Filter included actors
    sentences = filter_actor(sentences, actors)
    print("Actors included in data: {}".format(actors))

    # Filter senteces by emotions
    mapped_emotions = [MAPPING[e] for e in emotions]
    sentences = filter_emotion(mapped_emotions, sentences)
    print("Emotions included in data: {}".format(
        [list(MAPPING.keys())[list(MAPPING.values()).index(e)] for e in mapped_emotions]))

    # Get all frames from selected sentences
    all_paths = []
    for s in sorted(sentences):
        paths = glob(s + '/*.png')
        random.shuffle(paths)
        paths = paths[:max_frames_per_sentence] if max_frames_per_sentence else paths
        all_paths.append(sorted(paths))

    if flat:
        all_paths = [item for sublist in all_paths for item in sublist]

    if shuffled:
        random.shuffle(all_paths)

    # Count number of frames for every emotion and split in train and val
    print(f'# paths in total: {len(all_paths)}')
    train_paths, val_paths = [], []
    for emo in emotions:
        if flat:
            emo_lst = list(filter(lambda s: s.split('/')[-2].split('-')[2]
                                  in MAPPING[emo], all_paths))
        else:
            emo_lst = list(filter(lambda s: s[0].split('/')[-2].split('-')[2]
                                  in MAPPING[emo], all_paths))
        split = int(np.floor(validation_split * len(emo_lst)))
        train_paths = train_paths + emo_lst[split:]
        val_paths = val_paths + emo_lst[:split]

    return train_paths, val_paths, all_paths


def ravdess_get_paths_actor_split(root_path,
                                  flat,
                                  shuffled=True,
                                  validation_split=0.0,
                                  actors=[i + 1 for i in range(24)]):

    if root_path[-1] != '/':
        root_path += '/'

    paths = glob(root_path + '*/*/')
    split = int(len(actors) * validation_split)
    train_actors = actors[split:]
    val_actors = actors[:split]

    train_paths = list(
        filter(lambda s: int(s.split('/')[-3].split('_')[-1]) in train_actors, paths))
    val_paths = list(
        filter(lambda s: int(s.split('/')[-3].split('_')[-1]) in val_actors, paths))

    train_paths = [glob(p + '*.png') for p in train_paths]
    val_paths = [glob(p + '*.png') for p in val_paths]

    if flat:
        train_paths = [item for sublist in train_paths for item in sublist]
        val_paths = [item for sublist in val_paths for item in sublist]

    if shuffled:
        random.shuffle(train_paths)
        random.shuffle(val_paths)

    return train_paths, val_paths


def filter_actor(sentences, actors):
    return list(
        filter(lambda s: int(s.split('/')[-2].split('_')[-1]) in actors, sentences))


def filter_emotion(emotions, sentences):
    if type(sentences[0]) is list:
        return list(filter(lambda s: s[0].split('/')[-2].split('-')[2]
                    in emotions, sentences))
    else:
        return list(filter(lambda s: s.split('/')[-1].split('-')[2]
                    in emotions, sentences))


def int_to_one_hot(label, n_labels=8):
    one_hot = torch.zeros(n_labels)
    one_hot[label] = 1
    return one_hot


def omg_get_paths(root_path, flat=False, shuffled=False):
    root_dir = pathlib.Path(root_path)

    info = pd.read_csv(list(root_dir.glob('*_info.csv'))[0])

    videos = [str(v) for v in list(root_dir.glob('*/*/'))]

    all_paths = [sorted([str(p) for p in list(pathlib.Path(v).glob('*.png'))])
                 for v in videos]

    if flat:
        all_paths = [item for sublist in all_paths for item in sublist]

    if shuffled:
        random.shuffle(all_paths)

    return all_paths, info


def aff_wild_get_paths(root_path, flat=False, shuffled=False):
    if root_path[-1] != '/':
        root_path += ['/']
    root_dir = pathlib.Path(root_path)

    videos = [str(v) for v in list(root_dir.glob('*/'))
              if str(v).split('/')[-1] != 'annotations']

    all_paths = [sorted([str(p) for p in list(pathlib.Path(v).glob('*.png'))], key=lambda x: int(x.split('/')[-1].split('.')[0]))
                 for v in videos]

    annotations = {}
    for p in all_paths:
        filename = '/'.join(p[0].split('/')[:-2] + ['annotations'] + [p[0].split('/')[-2]]) + '.txt'
        # annotations.append([line.rstrip('\n').split(',') for line in open(filename)])
        anns = pd.read_csv(filename)
        valence = torch.tensor(anns['valence'].to_numpy(), dtype=torch.float32)
        arousal = torch.tensor(anns['arousal'].to_numpy(), dtype=torch.float32)
        a = torch.stack((valence, arousal), dim=1)
        annotations[p[0].split('/')[-2]] = a

    if flat:
        all_paths = [item for sublist in all_paths for item in sublist]

    if shuffled:
        random.shuffle(all_paths)

    return all_paths, annotations


def tagesschau_get_paths(root_path, train_split=1.0, max_frames_per_vid=-1):
    if root_path[-1] != '/':
        root_path += '/'
    # videos = glob(root_path + 'TV*/')
    videos = glob(root_path + 'sequence*/') + glob(root_path + 'TV*/')
    # videos = glob(root_path + 'sequence*/') + glob(root_path + 'TV*/') + glob(root_path + 'yt*/')
    # videos = glob(root_path + 'yt*/')
    # videos = glob(root_path + '*/')
    random.shuffle(videos)
    split = int(len(videos) * train_split)
    train_videos = videos[:split]
    val_videos = videos[split:]
    train_paths = [sorted([p.split('.')[0] for p in glob(v + '*.png')])[:max_frames_per_vid]
                   for v in train_videos]
    val_paths = [sorted([p.split('.')[0] for p in glob(v + '*.png')])[:max_frames_per_vid]
                 for v in val_videos]
    return train_paths, val_paths


def ffhq_get_paths(root_path, train_split=1.0):
    paths = glob(root_path + '*.png')
    random.shuffle(paths)
    split = int(len(paths) * train_split)
    train_paths = paths[:split]
    val_paths = paths[split:]
    return train_paths, val_paths


class Downsample(object):
    """ Custom transform: Downsamples image in StyleGAN2 manner """

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


class RandomSequenceSampler(Sampler):
    """
    Samples sequences of indices

    example usage:
        sampler = RandomSequenceSampler(range(len(ds)), 4)

    args:
        data_source: iterable
        sequence_length: int
    """

    def __init__(self, data_source, sequence_length):
        l_batched = []
        for i in range(0, len(data_source) - sequence_length + 1):
            l_batched.append(data_source[i:i + sequence_length])

        self.l_batched = torch.tensor(l_batched)
        print(self.l_batched)

    def __iter__(self):
        l_batched = self.l_batched[torch.randperm(
            len(self.l_batched))].tolist()
        return iter([item for sublist in l_batched for item in sublist])

    def __len__(self):
        return len(self.data_source)


class RandomTagesschauAudioSampler(Sampler):
    """
    Samples batches of sequential indices of length T + 1 (last index is for
    random input frame). 
    If weighted, the probability a video is chosen depends on its length.

    example usage:
        sampler = RandomTagesschauAudioSampler(paths, T=8, batch_size=args.batch_size)

    args:
        paths (list of lists):
        T (int):
        batch_size (int):
        num_batches (int):
        weighted (bool):
    """

    def __init__(self, paths, T, batch_size, num_batches, weighted=False):
        indices = []
        i = 0
        for path in paths:
            indices.append([])
            for p in path:
                indices[-1].append(i)
                i += 1
        self.indices = indices
        self.T = T
        self.batch_size = batch_size
        self.num_batches = num_batches
        if weighted:
            len_videos = [len(v) for v in indices]
            self.prob_video = [float(length) / sum(len_videos)
                               for length in len_videos]
        else:
            self.prob_video = [1. / len(self.indices) for _ in self.indices]

    def __iter__(self):
        batch = []
        videos = random.choices(self.indices, weights=self.prob_video, k=len(self))
        for video in videos:
            start = random.randint(0, len(video) - self.T)
            inp_idx = random.choice(video)
            sample = video[start: start + self.T] + [inp_idx]
            batch.append(sample)
        return iter(batch)

    def __len__(self):
        return self.batch_size * self.num_batches
