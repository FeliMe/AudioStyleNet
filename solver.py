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
from torchvision.utils import make_grid

import utils


class ClassificationSolver:
    def __init__(self, config):

        self.save_dir = 'saves'
        self.config = config
        self.device = 'cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu'
        self.loss_names = ['bce_loss']
        self.model = config.model.train().to(self.device)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        if config.log_run:
            wandb.watch(self.model)

        print("Training on {}".format(self.device))
        print("Model type: {}".format(type(self.model)))

        self.optimizer = torch.optim.Adam(params=config.model.parameters(),
                                          lr=config.learning_rate)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1)

        self.x = None
        self.y = None
        self.logits = None
        self.y_ = None
        self.loss = None
        self.running_loss = None
        self.running_corrects = None

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
        Unpack input data from the dataloader and perform necessary pre-processing steps

        args:
            inputs (dict): includes the data itself and its metadata information
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
        self.loss = self.criterion(self.logits, self.y)

        if phase == 'train':
            self.loss.backward()
            self.optimizer.step()

    def update_statistics(self):
        self.running_loss += self.loss.item() * self.x.size(0)
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

                    # forward
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


class GANSolver(object):
    def __init__(self, generator, discriminator, classifier, mean, std):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
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
        print('Saving models to {}'.format(self.save_path))
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

        # Make grid image
        img_sample = torch.cat((real_a, fake_b, real_b), -2)
        img_sample = make_grid(img_sample, nrow=real_a.size(0), normalize=True)

        return img_sample

    def train_model(self,
                    optimizer_g,
                    optimizer_d,
                    criterion_gan,
                    criterion_pix,
                    criterion_emotion,
                    device,
                    train_loader,
                    val_loader,
                    config,
                    plot_grads=False,
                    log_run=True,
                    tb_writer=None):

        self.config = config
        self.log_run = log_run
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.save_path = os.path.join(
            config.save_path,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        grad_plotter_g, grad_plotter_d = None, None

        step = 0

        # Initialize vs
        v_gan, v_pixel, v_emotion = 0, 0, 0

        print("Starting training")
        t_start = time.time()

        for i_epoch in range(1, config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, config.num_epochs))
            print('-' * 10)

            epoch_acc_real = 0.
            epoch_acc_fake = 0.

            for batch in train_loader:

                # Increment step counter
                step += 1

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

                # Emotion loss
                embedding_fake = self.classifier(fake_b)
                embedding_real = self.classifier(real_b)
                emotion_loss = criterion_emotion(embedding_fake, embedding_real)

                # Adapt losses
                # ada_gan_loss, v_gan = utils.ada_loss(gan_loss, v_gan)
                # ada_pixel_loss, v_pixel = utils.ada_loss(pixel_loss, v_pixel)
                # ada_emotion_loss, v_emotion = utils.ada_loss(emotion_loss, v_emotion)

                # Total loss
                # generator_loss = ada_gan_loss + ada_pixel_loss + ada_emotion_loss
                generator_loss = gan_loss + config.lambda_pixel * pixel_loss + \
                    config.lambda_emotion * emotion_loss

                generator_loss.backward()

                optimizer_g.step()

                # v_gan = v_gan.detach()
                # v_pixel = v_pixel.detach()
                # v_emotion = v_emotion.detach()

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

                # Don't train discriminator if already too good
                if discriminator_loss.item() > 0.3:
                    discriminator_loss.backward()
                    # torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), 0.6)
                    optimizer_d.step()

                # Get mean anyway for logging
                # gan_loss = gan_loss.mean()
                # pixel_loss = pixel_loss.mean()
                # emotion_loss = emotion_loss.mean()

                # Plot gradients
                if plot_grads:
                    # Plot generator gradients
                    if grad_plotter_g is None:
                        grad_plotter_g = utils.GradPlotter(self.generator.named_parameters())
                    grad_plotter_g.plot_grad_flow(self.generator.named_parameters())

                    # Plot discriminator gradients
                    if grad_plotter_d is None:
                        grad_plotter_d = utils.GradPlotter(self.discriminator.named_parameters())
                    grad_plotter_d.plot_grad_flow(self.discriminator.named_parameters())

                # Accuracy of discriminator
                acc_real = pred_real.round() == valid_patch.view(pred_real.size()).data
                acc_fake = pred_fake.round() == fake_patch.view(pred_fake.size()).data
                acc_real = acc_real.double().mean()
                acc_fake = acc_fake.double().mean()
                epoch_acc_real += acc_real
                epoch_acc_fake += acc_fake

                # W&B Logging
                if self.log_run:
                    tb_writer.add_scalars(
                        'generator/',
                        {
                            'GAN': gan_loss,
                            'pixelwise': pixel_loss,
                            'emotion': emotion_loss
                        },
                        step)
                    tb_writer.add_scalar('discriminator/loss',
                                         discriminator_loss, step)
                    tb_writer.add_scalar('discriminator/acc_fake',
                                         acc_fake, step)
                    tb_writer.add_scalar('discriminator/acc_real',
                                         acc_real, step)

            # --------------
            #  Log Progress
            # --------------

            epoch_acc_real /= len(train_loader)
            epoch_acc_fake /= len(train_loader)

            print('Generator GAN loss: {:.4f}'.format(gan_loss))
            print('Generator pixelwise loss: {:.4f}'.format(pixel_loss))
            print('Generator emotion loss: {:.4f}'.format(emotion_loss))
            print('Discriminator loss: {:.4f}'.format(discriminator_loss))
            print('Discriminator acc real: {:.4f}'.format(epoch_acc_real))
            print('Discriminator acc fake: {:.4f}'.format(epoch_acc_fake))
            print('Time elapsed {}'.format(utils.time_to_str(time.time() - t_start)))
            print('Time left: {}\n'.format(
                utils.time_left(t_start, config.num_epochs, i_epoch)))

            if log_run:
                # Get sample from validation set
                val_batch = next(iter(val_loader))
                val_a = val_batch['A'].to(device)
                val_b = val_batch['B'].to(device)

                # Generate sample images
                img_sample = self.sample_images(val_a, val_b, i_epoch)
                tb_writer.add_image('sample', img_sample, i_epoch)

                # Save model
                self.save()

        time_elapsed = time.time() - t_start
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        if self.log_run:
            self.save()
