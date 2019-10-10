import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time
import torch
import wandb

from sklearn.metrics import confusion_matrix


class Solver(object):
    def __init__(self,
                 model):
        self.model = model
        self.config = None
        self.save_path = 'saves'

    def train_model(self,
                    criterion,
                    optimizer,
                    device,
                    data_loaders,
                    dataset_sizes,
                    config,
                    scheduler=None):
        self.config = config
        self.save_path = os.path.join(
            config.save_path,
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
                            # wandb.log({'Gradients': wandb.Histogram(
                            #     self.model.named_parameters())})

                    if (i_step + 1) % config.log_interval == 0:
                        print("Step {}/{}".format(
                            i_step + 1, int(dataset_sizes[phase] / config.batch_size)))

                    # statistics
                    running_loss += loss.item() * x.size(0)
                    running_corrects += torch.sum(y_ == y.data)
                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # W&B logging
                wandb.log({phase + ' Loss': epoch_loss,
                           phase + ' Accuracy': epoch_acc})

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
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pt'))
        return self.model

    def eval_model(self,
                   device,
                   data_loaders,
                   num_eval=10000):
        y_true = []
        y_pred = []
        for i_stop, (x, y) in enumerate(data_loaders['val']):
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
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm,
            annot=True,
            xticklabels=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
            yticklabels=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
            linewidths=.5,
            fmt=".2f",
            square=True,
            cmap='Blues',
        )
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        os.makedirs(self.save_path, exist_ok=True)
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix.jpg'))
        wandb.log({'confusion matrix': [wandb.Image(plt)]})


def plot_grad_flow(named_parameters):
    """
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    'plot_grad_flow(self.model.named_parameters())' to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max([tensor.cpu() for tensor in ave_grads]))  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
