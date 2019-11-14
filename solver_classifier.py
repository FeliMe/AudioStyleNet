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

import utils


class ClassificationSolver:
    def __init__(self, config):

        # General
        self.save_dir = 'saves'
        self.config = config
        self.device = 'cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu'
        print("Training on {}".format(self.device))

        # Models
        self.model = config.model.train().to(self.device)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        if config.log_run:
            wandb.watch(self.model)

        print("Model type: {}".format(type(self.model)))
        print("# params {} (trainable {})".format(
            utils.count_params(self.model),
            utils.count_trainable_params(self.model)
        ))

        # Optimizer
        self.optimizer = torch.optim.Adam(params=config.model.parameters(),
                                          lr=config.learning_rate)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Losses
        self.loss_names = ['bce_loss']
        self.loss_classifier_bce = None
        self.running_loss = 0
        self.running_corrects = 0

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1)

        self.logits = None
        self.y_ = None

    def save(self):
        """
        Save the best model to self.save_dir
        """
        self.model.load_state_dict(self.best_model_wts)
        os.makedirs(self.save_dir, exist_ok=True)
        print('Saving model to {}'.format(
            os.path.join(self.save_dir, 'best_model.pt')))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, 'best_model.pt'))
        if self.config.log_run:
            torch.save(self.model.state_dict(),
                       os.path.join(wandb.run.dir, 'best_model.pt'))

    def plot_grads(self, grad_plotter):
        """
        Plots the gradients in every iteration or initializes the grad_plotter
        when first calles
        args:
            grad_plotter (GradPlotter): object which updates figure with gradients
                                        on every iteration
        """
        if grad_plotter is None:
            grad_plotter = utils.GradPlotter(self.model.named_parameters())
        grad_plotter.plot_grad_flow(self.model.named_parameters())

    def set_input(self, inputs):
        """
        Unpack input data from the dataloader

        args:
            inputs (dict): packaged input data
        """
        self.x = inputs['x'].to(self.device)
        self.y = inputs['y'].to(self.device)

    def forward(self):
        """
        Run forward pass
        """
        self.logits = self.model(self.x)

    def optimize_parameters(self, phase):
        """
        Calculate losses, gradients, and update network weights
        """
        self.forward()
        self.optimizer.zero_grad()
        self.loss_classifier_bce = self.criterion(self.logits, self.y)

        if phase == 'train':
            self.loss_classifier_bce.backward()
            self.optimizer.step()

    def update_statistics(self):
        self.running_loss += self.loss_classifier_bce.item() * self.x.size(0)
        _, self.y_ = torch.max(self.logits, 1)
        self.running_corrects += torch.sum(self.y_ == self.y.data)

    def train_model(self,
                    data_loaders,
                    dataset_sizes,
                    tb_writer,
                    plot_grads=False):
        """
        Run training of the classification model

        args:
            data_loaders (dict of torch.utils.data.Datasets): Iterable data_loaders
            dataset_sizes (dict of ints): Number of samples in each data_loader
            tb_writer (torch.utils.tensorboard.SummaryWriter): Tensorboard writer
            plot_grads (Bool): Plot gradient flow or not
        """

        self.save_dir = os.path.join(
            self.config.save_path,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        print("Starting training")

        t_start = time.time()
        grad_plotter = None
        best_acc = 0.0
        global_step = 0

        for i_epoch in range(1, self.config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, self.config.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                self.running_loss = 0.0
                self.running_corrects = 0

                # Iterate over data.
                for batch in data_loaders[phase]:

                    self.set_input(batch)

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        self.optimize_parameters(phase)
                        if phase == 'train':
                            if plot_grads:
                                self.plot_grads(grad_plotter)

                    # statistics
                    self.update_statistics()

                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss = self.running_loss / dataset_sizes[phase]
                epoch_acc = self.running_corrects.double() / dataset_sizes[phase]

                # Logging
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val':
                    time_elapsed = time.time() - t_start
                    print('Time elapsed {}'.format(
                        utils.time_to_str(time_elapsed)))
                    print('Time left: {}\n'.format(
                        utils.time_left(t_start, self.config.num_epochs, i_epoch)))

                # W&B logging
                if self.config.log_run:
                    tb_writer.add_scalar(phase + '/Loss',
                                         epoch_loss, global_step)
                    tb_writer.add_scalar(phase + '/Accuracy',
                                         epoch_acc, global_step)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - t_start
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.save()
        return self.model

    def eval_model(self,
                   data_loaders,
                   tb_writer,
                   num_eval=10000):
        y_true = []
        y_pred = []
        for i_stop, (x, y) in enumerate(data_loaders['val']):
            x = x.to(self.device)
            y = y.to(self.device)

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
        f = plt.figure(figsize=(8, 8))
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
        os.makedirs(self.save_dir, exist_ok=True)
        f.savefig(os.path.join(self.save_dir, 'confusion_matrix.jpg'))
        if self.config.log_run:
            tb_writer.add_figure('confusion matrix', f)