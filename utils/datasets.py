"""
File to specify dataloaders for different datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import torch
import torch.nn.functional as F

from my_models.models import FERClassifier
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import make_grid


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
        seed (int): Random seed for reproducible shuffling
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
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 seed=123,
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
                 normalize=False,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 image_size=256,
                 label_one_hot=False):

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

        return {'x': img, 'y': emotion, 'index': index, 'paths': frame}


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
        sentences_neutral = filter_emotion([MAPPING['neutral']], all_paths)
        sentences_x = filter_emotion(MAPPING[self.emotion_x], paths)

        # Get x_neutral pairs
        sentence_pairs = {'neutral': [], 'x': []}
        for s in sentences_x:
            sentence_pairs['x'].append(s)
            ident = s[0].split('/')[-2].split('-')
            ident[2] = MAPPING['neutral']
            ident[3] = '01'
            ident = '-'.join(ident)
            sentence_pairs['neutral'] += list(
                filter(lambda s: s[0].split('/')[-2] == ident, sentences_neutral))
        #
        # sentences_neutral = [sentences_neutral[0]]
        # sentences_x = [sentences_x[0]]
        #
        # Get length of sentences
        len_sentences_neutral = [len(s) for s in sentence_pairs['neutral']]
        len_sentences_x = [len(s) for s in sentence_pairs['x']]

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

        self.sentence_pairs = sentence_pairs
        self.len_sentences_neutral = len_sentences_neutral
        self.len_sentences_x = len_sentences_x

        self.scores_x = self._load_scores_x()

    def __len__(self):
        return len(self.sentence_pairs['x'])

    def _load_scores_x(self):
        print("Loading scores for {}".format(self.emotion_x))
        scores_x = []
        for s in self.sentence_pairs['x']:
            scores_x.append([])
            for f in s:
                emotion = int(f.split('/')[-2].split('-')[2]) - 1
                score_path = f.split('.')[0] + '-logit_{}.pt'.format(self.score_type)
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
        sentence_neutral = self.sentence_pairs['neutral'][index]
        sentence_x = self.sentence_pairs['x'][index]

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
                 device,
                 seed=123,
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

        # Filter in neutral sentences and rest
        neutral_sentences = filter_emotion(MAPPING['neutral'], paths)
        x_sentences = filter_emotion(
            [MAPPING[e] for e in emotions if e != 'neutral'], paths)

        # Get lengths of sentences
        len_neutral_sentences = [len(s) for s in neutral_sentences]
        len_x_sentences = [len(s) for s in x_sentences]

        # Transforms
        if int(np.log2(image_size)) - np.log2(image_size) == 0:
            trans = [transforms.ToTensor(), Downsample(image_size)]
        else:
            trans = [transforms.Resize(image_size), transforms.ToTensor()]
        if self.normalize:
            trans.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.t = transforms.Compose(trans)

        self.neutral_sentences = neutral_sentences
        self.x_sentences = x_sentences
        self.len_neutral_sentences = len_neutral_sentences
        self.len_x_sentences = len_x_sentences
        self.emotions = emotions

        print(len(self.neutral_sentences))
        print(len(self.x_sentences))

        # self.x_scores = self._compute_x_scores()

    def __len__(self):
        return len(self.sentence_pairs['x'])

    def __getitem__(self, index):
        # Select src and target sentence
        neutral_sentence = self.neutral_sentences[index]
        x_index = random.randint(0, len(self.x_sentences) - 1)
        x_sentence = self.x_sentences[x_index]

        # Get random index
        rand_neutral_idx = random.randint(0, self.len_neutral_sentences[index] - 1)
        rand_x_idx = min(rand_neutral_idx, self.x_sentences[x_index] - 1)

        # Load Images
        neutral = self.t(Image.open(neutral_sentence[rand_neutral_idx]))
        x = self.t(Image.open(x_sentence[rand_x_idx]))

        res = {
            'neutral': neutral,
            'x': x,
            'index': index,
            'rand_target_idx': rand_x_idx,
        }

        res['x_score'] = self.x_scores[x_index][rand_x_idx]

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


def get_paths(root_path,
              flat,
              shuffled=True,
              validation_split=0.0,
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
    all_paths = sorted([sorted([str(p) for p in list(pathlib.Path(s).glob('*.png'))])
                        for s in sentences])
    if flat:
        all_paths = [item for sublist in all_paths for item in sublist]

    if shuffled:
        # Shuffle sentences
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


def get_pairs(root_path,
              validation_split=0.0,
              emotions=['neutral', 'calm', 'happy', 'sad', 'angry',
                        'fearful', 'disgust', 'surprised'],
              actors=[i + 1 for i in range(24)]):
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
    mapped_emotions = [MAPPING[e] for e in emotions]
    sentences = filter_emotion(mapped_emotions, sentences)
    print("Emotions included in data: {}".format(
        [list(MAPPING.keys())[list(MAPPING.values()).index(e)] for e in mapped_emotions]))

    # Get all frames from selected sentences
    paths = sorted([sorted([str(p) for p in list(pathlib.Path(s).glob('*.png'))])
                    for s in sentences])

    # Get target_sentence pairs
    without_neutral = filter_emotion(
        [MAPPING[e] for e in emotions if e != 'neutral'], paths)
    sentence_pairs = {'neutral': [], 'x': []}
    for s in without_neutral:
        sentence_pairs['x'].append(s)
        ident = s[0].split('/')[-2].split('-')
        ident[2] = MAPPING['neutral']
        ident[3] = '01'
        ident = '-'.join(ident)
        sentence_pairs['neutral'] += list(
            filter(lambda s: s[0].split('/')[-2] == ident, paths))

    # Count number of pairs and split in train and val
    print('# pairs in total: {}'.format(len(sentence_pairs['x'])))

    # Split in train and val
    random.shuffle(sentence_pairs['x'])
    random.shuffle(sentence_pairs['neutral'])

    train_pairs = {'neutral': [], 'x': []}
    val_pairs = {'neutral': [], 'x': []}
    split = int(np.floor(validation_split * len(sentence_pairs['x'])))
    train_pairs['neutral'] = sentence_pairs['neutral'][split:]
    train_pairs['x'] = sentence_pairs['x'][split:]
    val_pairs['neutral'] = sentence_pairs['neutral'][:split]
    val_pairs['x'] = sentence_pairs['x'][:split]

    return train_pairs, val_pairs


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


def int_to_one_hot(label):
    one_hot = torch.zeros(8)
    one_hot[label] = 1
    return one_hot


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
