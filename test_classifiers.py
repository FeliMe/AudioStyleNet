import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.datasets import RAVDESSFlatDataset, ravdess_get_paths
from my_models import models


def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    return cm_normalized


def plot_confusion_matrix(cm, path, title='Normalized Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(14, 14))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=['neutral', 'happy', 'sad', 'angry',
                     'fearful', 'disgust', 'surprised'],
        yticklabels=['neutral', 'happy', 'sad', 'angry',
                     'fearful', 'disgust', 'surprised'],
        fmt=".3f",
        linewidths=.5,
        square=True,
        cmap='Blues',
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.savefig(path)


if __name__ == '__main__':

    # Random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Select model
    if args.model == 'fer':
        model = models.FERClassifier()
        # FER model doesn't predict calmness
        emotions = ['neutral', 'happy', 'sad', 'angry',
                    'fearful', 'disgust', 'surprised']
    elif args.model == 'ravdess':
        model = models.EmotionClassifier()
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry',
                    'fearful', 'disgust', 'surprised']
    else:
        raise NotImplementedError
    model = model.eval().to(device)

    # Prepare data
    ds_path = os.path.join(os.path.expanduser('~'), 'Datasets/RAVDESS/Aligned256/')
    paths, _ = ravdess_get_paths(
        root_path=ds_path,
        flat=True,
        emotions=emotions
    )
    ds = RAVDESSFlatDataset(
        paths=paths,
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        image_size=48,
        device=device
    )
    data_loader = DataLoader(ds, batch_size=32, drop_last=False, num_workers=4, shuffle=True)

    n_corrects = 0
    n_samples = len(ds)

    y_true, y_pred = None, None

    for i, batch in enumerate(tqdm(data_loader)):
        img = batch['x'].to(device)
        target_emotion_int = batch['y'].to(device)

        pred_emotion = model(img)
        pred_emotion_int = pred_emotion.max(dim=1)[1]

        n_corrects += (pred_emotion_int == target_emotion_int).sum()
        if y_true is None:
            y_true = target_emotion_int
            y_pred = pred_emotion_int
        else:
            y_true = torch.cat((y_true, target_emotion_int), dim=0)
            y_pred = torch.cat((y_pred, pred_emotion_int), dim=0)

    acc = n_corrects.item() / n_samples
    print("Accuracy: {:.4f}".format(acc))

    cm = make_confusion_matrix(y_true, y_pred)
    print(cm)

    os.makedirs('saves/test_classifiers/', exist_ok=True)
    plot_confusion_matrix(
        cm,
        'saves/test_classifiers/{}confusion_matrix.png'.format(args.model),
        title='Fer model github on RAVDESS. Accuracy: {:.4f}'.format(acc)
    )
