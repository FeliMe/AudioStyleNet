import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import signal
import time
import torch
import wandb

from sklearn.metrics import confusion_matrix
from torchvision.utils import save_image

import utils


class BaseSolver(object):
    def __init__(self, model):
        self.model = model
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.save_path = 'saves'
        self.config = None
        self.log_run = False
        self.kill_now = False

    def save(self):
        self.model.load_state_dict(self.best_model_wts)
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pt'))
        if self.log_run:
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))

    def exit_gracefully(self, signum, frame):
        print("Stopping gracefully, saving best model...")
        self.save()
        self.kill_now = True


class ClassificationSolver(BaseSolver):
    def __init__(self, model):
        super(ClassificationSolver, self).__init__(model)

    def train_model(self,
                    criterion,
                    optimizer,
                    device,
                    data_loaders,
                    dataset_sizes,
                    config,
                    scheduler=None,
                    plot_grads=False,
                    log_run=True):

        self.config = config
        self.log_run = log_run
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.save_path = os.path.join(
            config.save_path,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        grad_plotter = None

        print("Starting training")
        t_start = time.time()

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
                i_step = 0

                # Iterate over data.
                for x, y in data_loaders[phase]:
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
                            i_step += 1
                            loss.backward()
                            optimizer.step()

                            if plot_grads:
                                if grad_plotter is None:
                                    grad_plotter = utils.GradPlotter(self.model.named_parameters())
                                grad_plotter.plot_grad_flow(self.model.named_parameters())

                    if (i_step + 1) % config.log_interval == 0:
                        print("Step {}/{}".format(i_step + 1,
                                                  len(data_loaders[phase])))

                    # statistics
                    running_loss += loss.item() * x.size(0)
                    running_corrects += torch.sum(y_ == y.data)

                    if self.kill_now:
                        return self.model

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Logging
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val':
                    time_elapsed = time.time() - t_start
                    print('Time elapsed {}'.format(
                        utils.time_to_str(time_elapsed)))
                    print('Time left: {}\n'.format(
                        utils.time_left(t_start, config.num_epochs, i_epoch)))

                # W&B logging
                if self.log_run:
                    wandb.log({phase + ' Loss': epoch_loss,
                               phase + ' Accuracy': epoch_acc})

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
        if self.log_run:
            wandb.log({'confusion matrix': [wandb.Image(plt)]})


class GANSolver(object):
    def __init__(self, generator, discriminator, mean, std):
        self.generator = generator
        self.discriminator = discriminator
        self.save_path = 'saves'
        self.config = None
        self.log_run = False
        self.kill_now = False
        self.mean = mean
        self.std = std

    def exit_gracefully(self, signum, frame):
        print("Stopping gracefully, saving best model...")
        self.save()
        self.kill_now = True

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.generator.state_dict(),
                   os.path.join(self.save_path, 'generator.pt'))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.save_path, 'discriminator.pt'))
        if self.log_run:
            torch.save(self.generator.state_dict(),
                       os.path.join(wandb.run.dir, 'generator.pt'))
            torch.save(self.discriminator.state_dict(),
                       os.path.join(wandb.run.dir, 'discriminator.pt'))

    def sample_images(self, real_a, real_b, i_epoch):
        """ Saves a generated sample """

        # Generate fake sequence
        fake_b = self.generator(real_a[0].unsqueeze(0))

        # Denormalize one sequence
        transform = utils.denormalize(self.mean, self.std)
        real_a = torch.stack([transform(img) for img in real_a[0].data], dim=0)
        real_b = torch.stack([transform(img) for img in real_b[0].data], dim=0)
        fake_b = torch.stack([transform(img) for img in fake_b[0].data], dim=0)

        # Specify and create target folder
        target_dir = os.path.join(self.save_path, 'images')
        os.makedirs(target_dir, exist_ok=True)

        # Save
        img_sample = torch.cat((real_a, fake_b, real_b), -2)
        save_image(img_sample, target_dir + "/{}.png".format(i_epoch),
                   nrow=real_a.size(0), normalize=True)

    def train_model(self,
                    optimizer_g,
                    optimizer_d,
                    criterion_gan,
                    criterion_pix,
                    device,
                    data_loader,
                    config,
                    lambda_pixel=100,
                    plot_grads=False,
                    log_run=True):

        self.config = config
        self.log_run = log_run
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.save_path = os.path.join(
            config.save_path,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        grad_plotter = None

        print("Starting training")
        t_start = time.time()

        for i_epoch in range(1, config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, config.num_epochs))
            print('-' * 10)

            for batch in data_loader:

                # Inputs
                real_a = batch['A'].to(device)
                real_b = batch['B'].to(device)

                # ------------------
                #  Train Generator
                # ------------------

                optimizer_g.zero_grad()

                # Generate fake
                fake_b = self.generator(real_a)

                # Discriminate fake
                pred_fake, patch_size = self.discriminator(fake_b, real_a)
                valid_patch = torch.ones(patch_size, device=device)

                # GAN loss
                gan_loss = criterion_gan(pred_fake, valid_patch)

                # Pixel-wise loss
                pixel_loss = criterion_pix(fake_b, real_b)

                # Total loss
                generator_loss = gan_loss + lambda_pixel * pixel_loss

                generator_loss.backward()

                optimizer_g.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_d.zero_grad()

                # Train with all-real batch
                fake_patch = torch.zeros(patch_size, device=device)
                pred_real, _ = self.discriminator(real_b, real_a)
                loss_real = criterion_gan(pred_real, valid_patch)

                # Train with all-fake batch
                pred_fake, _ = self.discriminator(fake_b.detach(), real_a)
                loss_fake = criterion_gan(pred_fake, fake_patch)

                # Total loss
                discriminator_loss = 0.5 * (loss_real + loss_fake)

                discriminator_loss.backward()
                optimizer_d.step()

                # W&B Logging
                if self.log_run:
                    wandb.log({'Generator loss': generator_loss,
                               'Discriminator loss': discriminator_loss})

            # --------------
            #  Log Progress
            # --------------

            print('Generator loss: {:.4f} Discriminator loss: {:.4f}'.format(
                generator_loss, discriminator_loss))
            time_elapsed = time.time() - t_start
            print('Time elapsed {}'.format(
                utils.time_to_str(time_elapsed)))
            print('Time left: {}\n'.format(
                utils.time_left(t_start, config.num_epochs, i_epoch)))

            self.sample_images(real_a, real_b, i_epoch)
            if log_run:
                self.save()

        time_elapsed = time.time() - t_start
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        if self.log_run:
            self.save()
