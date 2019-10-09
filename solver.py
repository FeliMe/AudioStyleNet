import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time
import torch

from sklearn.metrics import confusion_matrix


class Solver(object):
    def __init__(self,
                 model):
        self.model = model
        self.config = None
        self.save_path = './saves'

    def train_model(self,
                    criterion,
                    optimizer,
                    scheduler,
                    device,
                    data_loaders,
                    dataset_sizes,
                    config):
        self.config = config
        self.save_path = os.path.join(
            self.config.save_path,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        print("Starting training")
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for i_epoch in range(1, config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, config.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i_step, (x, y) in enumerate(data_loaders[phase]):
                    x = x.to(device)
                    y = y.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(x)
                        _, y_ = torch.max(logits, 1)
                        loss = criterion(logits, y)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    if (i_step + 1) % config.log_interval == 0:
                        print("Step {}/{}".format(i_step + 1, dataset_sizes[phase]))

                    # statistics
                    running_loss += loss.item() * x.size(0)
                    running_corrects += torch.sum(y_ == y.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def eval_model(self,
                   device,
                   data_loaders,
                   num_eval=10000):
        y_true = []
        y_pred = []
        for i_stop, (x, y) in enumerate(data_loaders['val']):
            print(i_stop)
            x = x.to(device)
            y = y.to(device)

            logits = self.model(x)
            _, y_ = torch.max(logits, 1)

            y_true.append(y.cpu().numpy())
            y_pred.append(y_.cpu().numpy())

            if (i_stop + 1) % num_eval == 0:
                break

        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm,
            annot=True,
            xticklabels=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
            yticklabels=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
            linewidths=.5,
            square=True,
            cmap='Blues',
        )
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix.jpg'))
