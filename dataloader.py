"""
File to specify dataloaders for different datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import torch

from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import utils


class RAVDESSDataset(Dataset):
    """
    Dataset class for loading RAVDESS sentences in a sequential manner.

    Shortest sentence in RAVDESS has 94 frames.

    Output shapes:
        'image': [batch_size, 1 or 3, height, width]
        'landmarks': [batch_size, 68 * 2]

    Arguments:
        root_path (str): Path to data files
        data_format (str): Format of data files ('image' or 'landmarks')
        normalize (bool): Normalize data before outputting
        mean (list): Dataset mean values
        std (list): Dataset standard deviations
        max_samples (int or None): Maximum number of samples to be considered.
                                   Choose None for whole dataset
        seed (int): Random seed for reproducible shuffling
        image_size (int or tuple): Size of input images
        num_classes (int): Number of classes (used for conditioning)
        label_one_hot (bool): Choose if emotion is scalar or one_hot vector
        emotions (list of str): List of emotions to be considered
        actors (list of int): List of actors to be considered
    """
    def __init__(self,
                 root_path,
                 data_format='image',
                 normalize=True,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 max_samples=None,
                 seed=123,
                 image_size=64,
                 num_classes=8,
                 label_one_hot=False,
                 emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                           'fearful', 'disgust', 'surprised'],
                 actors=[i + 1 for i in range(24)]):

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
        sentences = list(
            filter(lambda s: int(s.split('/')[-2].split('_')[-1]) in actors, sentences))
        print("Actors included in data: {}".format(actors))

        # Filter senteces by emotions
        self.mapping = {
            'neutral': '01',
            'calm': '02',
            'happy': '03',
            'sad': '04',
            'angry': '05',
            'fearful': '06',
            'disgust': '07',
            'surprised': '08'
        }
        self.emotions = [self.mapping[e] for e in emotions]
        sentences = list(filter(lambda s: s.split('/')[-1].split('-')[2]
                                in self.emotions, sentences))
        print("Emotions included in data: {}".format(
            [list(self.mapping.keys())[list(self.mapping.values()).index(e)] for e in self.emotions]))

        # Get all frames from selected sentences
        frames = [str(f) for s in sentences for f in list(
            pathlib.Path(s).glob('*.jpg'))]
        frames += [str(f) for s in sentences for f in list(
            pathlib.Path(s).glob('*.png'))]

        # Count number of frames for every emotion
        tmp = [f.split('/')[-2].split('-')[2] for f in frames]
        for emo in emotions:
            print("# frames for '{}': {}".format(
                emo, len(list(filter(lambda t: t == self.mapping[emo], tmp)))))

        # Random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Shuffle frames
        # random.shuffle(sentences)
        random.shuffle(frames)

        if max_samples is not None:
            # sentences = sentences[:min(len(sentences), max_samples)]
            frames = frames[:min(len(frames), max_samples)]

        trans = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.transforms = transforms.Compose(trans)

        self.sentences = sentences
        self.frames = frames

        if data_format == 'image':
            self.load_fn = load_image
            self.show_fn = show_images
        elif data_format == 'landmarks':
            self.load_fn = load_landmark
            self.show_fn = show_landmarks
        else:
            raise (RuntimeError('Unknown format {}'.format(data_format)))

    # def _get_sample(self, sentence, idx):
    #     # Get paths to load
    #     path = os.path.join(sentence, str(idx).zfill(3) + '.jpg')
    #     x = self.load_fn(path, self.transforms)

    #     return x

    # def _get_random_idx(self, sentence):
    #     len_sentence = len(list(pathlib.Path(sentence).glob('*')))
    #     rand_idx = torch.randint(1, len_sentence + 1, (1,)).item()

    #     return rand_idx

    def _int_to_one_hot(self, label):
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot

    def show_sample(self):
        """
        Plot a random sample
        """
        sample, _ = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)

    def __len__(self):
        # return len(self.sentences)
        return len(self.frames)

    def __getitem__(self, item):
        # Select a sentence
        # sentence = self.sentences[item]

        # Get sample
        # x = self._get_sample(sentence, self._get_random_idx(sentence))

        # Load Frame
        frame = self.frames[item]
        x = self.load_fn(frame, self.transforms)

        # Get emotion
        # emotion = int(sentence.split('/')[-1].split('-')[2]) - 1
        emotion = int(frame.split('/')[-2].split('-')[2]) - 1
        if self.label_one_hot:
            emotion = self._int_to_one_hot(emotion)

        return {'x': x, 'y': emotion}


class RAVDESSDSPix2Pix(RAVDESSDataset):
    def __init__(self,
                 root_path,
                 target_root_path,
                 data_format='image',
                 use_same_sentence=True,
                 normalize=True,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 max_samples=None,
                 seed=999,
                 image_size=64,
                 num_classes=8,
                 label_one_hot=False,
                 emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                           'fearful', 'disgust', 'surprised'],
                 actors=[i + 1 for i in range(24)]):
        super(RAVDESSDSPix2Pix, self).__init__(root_path, data_format,
                                               normalize, mean, std,
                                               max_samples, seed,
                                               image_size, num_classes,
                                               label_one_hot, emotions, actors)

        self.target_root_path = target_root_path
        self.use_same_sentence = use_same_sentence
        self.show_fn = show_pix2pix

        if normalize:
            print("Mean: {}, std: {}".format(self.mean, self.std))
        else:
            print("No normalization")

    def __getitem__(self, item):
        """
        Gets a pair of sequences (input b and target a). Source of the sequences
        is defined by root_path for input and target_root_path for target
        """

        # # Input sentence
        # input_sentence = self.sentences[item]
        # indices = self._get_random_idx(input_sentence)
        # a = self._get_sample(input_sentence, indices)

        # # Target sentence
        # if self.use_same_sentence:
        #     target_sentence = os.path.join(
        #         self.target_root_path, *input_sentence.split('/')[-2:])
        # else:
        #     # Use sentence from same actor
        #     actor = os.path.join(self.target_root_path,
        #                          *input_sentence.split('/')[-2:-1])
        #     # Get all sentences from actor
        #     all_sentences = [str(p)
        #                      for p in list(pathlib.Path(actor).glob('*'))]
        #     # Target sentences must have different emotion
        #     inp_emotion = input_sentence.split('/')[-1].split('-')[2]
        #     emotions = [e for e in self.emotions if e != inp_emotion]
        #     all_sentences = list(filter(lambda s: s.split('/')[-1].split('-')[2]
        #                                 in emotions, all_sentences))
        #     # Randomly select a sentence
        #     target_sentence = random.choice(all_sentences)
        #     indices = self._get_random_idx(target_sentence)
        # b = self._get_sample(target_sentence, indices)

        # # Get emotion from target sentence
        # emotion = int(target_sentence.split('/')[-1].split('-')[2]) - 1
        # if self.label_one_hot:
        #     emotion = self._int_to_one_hot(emotion)

        # Input frame
        input_frame = self.frames[item]
        a = self.load_fn(input_frame, self.transforms)

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
        b = self.load_fn(target_frame, self.transforms)

        emotion = int(target_frame.split('/')[-2].split('-')[2]) - 1
        if self.label_one_hot:
            emotion = self._int_to_one_hot(emotion)

        return {'A': a, 'B': b, 'y': emotion, 'idx': item}

    def show_sample(self):
        """
        Plot a random sample
        """
        sample = self.__getitem__(np.random.randint(0, self.__len__() - 1))
        self.show_fn(sample, self.mean, self.std, self.normalize)


def load_images(paths, transform):
    x = []
    for path in paths:
        x.append(load_image(path, transform))
    return torch.stack(x, dim=0)


def load_image(path, transform):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        img = transform(img)
        return img


def load_landmarks(paths, transform):
    x = []
    for path in paths:
        x.append(load_landmark(path + '.npy', transform))
    return torch.cat(x, dim=0)


def load_landmark(path, transform):
    landmarks = torch.tensor(np.load(path), dtype=torch.float)
    return landmarks.reshape(-1)


def show_images(img, mean, std, normalize):
    """
    Plots a sequence of images
    """
    if normalize:
        transform = utils.denormalize(mean, std)
        img = torch.stack([transform(a) for a in img], 0)
    img = make_grid(img, nrow=img.size(0), normalize=True)
    plt.figure(figsize=(img.size(0), 1))
    plt.imshow(np.moveaxis(img.numpy(), 0, 2))
    plt.show()


def show_landmarks(landmarks, *kargs):
    if len(landmarks.shape) == 2:
        landmarks = landmarks[0]
    landmarks = landmarks[:2 * 68].reshape(-1, 2)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()


def show_pix2pix(sample, mean, std, normalize):
    """
    Plots a sample (input sequence and target sequence)
    """
    img_a = sample['A']
    img_b = sample['B']

    # Denormalize
    if normalize:
        transform = utils.denormalize(mean, std)
        img_a = transform(img_a)
        img_b = transform(img_b)
        # img_a = torch.stack([transform(a) for a in img_a], 0)
        # img_b = torch.stack([transform(b) for b in img_b], 0)

    # Make image grid
    imgs = torch.stack([img_a, img_b])
    imgs = make_grid(imgs, nrow=img_a.size(0), normalize=True)

    # Plot image
    plt.figure(figsize=(img_a.size(0), 2))
    plt.imshow(np.moveaxis(imgs.numpy(), 0, 2))
    plt.axis('off')
    plt.show()


def get_data_loaders(dataset, validation_split, batch_size, use_cuda):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = RandomSampler(train_indices)
    val_sampler = RandomSampler(val_indices)

    if use_cuda and torch.cuda.is_available():
        print("Pinning memory")
        kwargs = {'pin_memory': True}
    else:
        kwargs = {}

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              sampler=train_sampler,
                              drop_last=True,
                              **kwargs)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            sampler=val_sampler,
                            drop_last=True,
                            **kwargs)

    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }

    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices)
    }

    return data_loaders, dataset_sizes


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
        self.load_fn = load_image

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


class SimpleDataset(Dataset):
    def __init__(self,
                 root_path,
                 normalize=True,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 img_size=64,
                 seed=999):
        self.normalize = normalize
        self.mean = mean
        self.std = std

        root_dir = pathlib.Path(root_path)

        # Get paths to all sentences
        self.files = [str(p) for p in list(root_dir.glob('*'))
                      if str(p).split('/')[-1] != '.DS_Store']

        # Random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        trans = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ]
        if self.normalize:
            trans.append(transforms.Normalize(self.mean, self.std))

        self.trans = transforms.Compose(trans)
        self.load_fn = load_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.load_fn(self.files[item], self.trans)
