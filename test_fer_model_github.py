import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.dataloader import RAVDESSFlatDataset
from my_models.models import FERModelGitHub


def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    return cm_normalized


def plot_confusion_matrix(cm, title='Normalized Confusion matrix', cmap=plt.cm.Blues):
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
    plt.savefig('saves/fer_model_github_confusion_matrix.png')


# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare data
ds_path = os.path.join(os.path.expanduser('~'), 'Datasets/RAVDESS/Aligned256/')
ds = RAVDESSFlatDataset(
    root_path=ds_path,
    normalize=False,
    image_size=48,
    emotions=['neutral', 'happy', 'sad', 'angry',
              'fearful', 'disgust', 'surprised']
)
data_loader = DataLoader(ds, batch_size=32, drop_last=False, num_workers=4)
sample = next(iter(ds))

# Select model
model = FERModelGitHub(pretrained=True).eval().to(device)

n_corrects = 0
n_samples = len(ds)

y_true, y_pred = None, None

for i, batch in enumerate(tqdm(data_loader)):
    img = batch['x'].to(device)
    target_emotion_int = batch['y'].to(device)

    pred_emotion = F.softmax(model(img), dim=1)
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

plot_confusion_matrix(cm, title='Fer model github on RAVDESS. Accuracy: {:.4f}'.format(acc))
