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

from utils import utils


class ClassificationSolver:
    def __init__(self, config):

        # General
        self.save_dir = 'saves'
        self.config = config
        self.device = 'cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu'
        self.global_step = 0
        self.t_start = 0
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
        self.epoch_loss = 0.0
        self.epoch_acc = 0

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

    def log_tensorboard(self, tb_writer, phase):
        """
        Log metrics to tensorboard
        """
        tb_writer.add_scalar(phase + '/Loss',
                             self.epoch_loss, self.global_step)
        tb_writer.add_scalar(phase + '/Accuracy',
                             self.epoch_acc, self.global_step)

    def log_console(self, i_epoch, phase):
        """
        Log running metrics to console after each epoch

        args:
            i_epoch (int): Current epoch
        """
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, self.epoch_loss, self.epoch_acc))
        if phase == 'val':
            time_elapsed = time.time() - self.t_start
            print('Time elapsed {}'.format(
                utils.time_to_str(time_elapsed)))
            print('Time left: {}'.format(
                utils.time_left(self.t_start, self.config.num_epochs, i_epoch)))
            max_memory = torch.cuda.max_memory_allocated(self.device) / 1e6
            print("Max memory on {}: {} MB\n".format(self.device, int(max_memory)))

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

        # Update running metrics
        self.epoch_loss += self.loss_classifier_bce.item()
        _, self.y_ = torch.max(self.logits, 1)
        self.epoch_acc += torch.sum(self.y_ == self.y.data, dtype=torch.float) / self.y_.numel()

        if phase == 'train':
            self.loss_classifier_bce.backward()
            self.optimizer.step()

    def train_model(self,
                    data_loaders,
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

        self.t_start = time.time()
        grad_plotter = None
        best_acc = 0.0

        for i_epoch in range(1, self.config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, self.config.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                # Zero running metrics
                self.epoch_loss = 0.0
                self.epoch_acc = 0

                # Iterate over data.
                for batch in data_loaders[phase]:

                    self.global_step += 1

                    self.set_input(batch)

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        self.optimize_parameters(phase)
                        if phase == 'train':
                            if plot_grads:
                                self.plot_grads(grad_plotter)

                    # Tensorboard logging
                    if self.config.log_run:
                        self.log_tensorboard(tb_writer, phase)

                # ---------------
                #  Epoch finished
                # ---------------

                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()

                # Mean running metrics
                self.epoch_loss = self.epoch_loss / len(data_loaders[phase])
                self.epoch_acc = self.epoch_acc.double() / len(data_loaders[phase])

                # Logging
                self.log_console(i_epoch, phase)

                # deep copy the model
                if phase == 'val' and self.epoch_acc > best_acc:
                    best_acc = self.epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - self.t_start
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