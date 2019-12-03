import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import utils

from models.model_utils import weights_init


class GANSolver(object):
    def __init__(self, config):

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        # General
        self.config = config
        self.device = 'cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu'
        self.global_step = 0
        self.t_start = 0
        self.mapping_list = np.array(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise'])

        print("Training on {}".format(self.device))

        # Models
        self.model_names = ['generator', 'discriminator', 'classifier']
        self.generator = config.generator.to(self.device)
        self.generator.apply(weights_init)

        self.discriminator = config.discriminator.to(self.device)
        self.discriminator.apply(weights_init)

        self.classifier = config.classifier.train().to(self.device)
        self.classifier.load_state_dict(torch.load(config.classifier_path, map_location=self.device))
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Print model info
        for name in self.model_names:
            if isinstance(name, str):
                model = getattr(self, name)
                print("{}: # params {} (trainable {})".format(
                    name,
                    utils.count_params(model),
                    utils.count_trainable_params(model)
                ))

        # Optimizers
        self.optimizer_G = self.config.optimizer_G
        self.optimizer_D = self.config.optimizer_D

        # Loss Functions
        self.criterionGAN = utils.GANLoss(config.GAN_mode, self.device,
                                          flip_p=config.flip_prob,
                                          noisy_labels=config.noisy_labels,
                                          label_range_real=config.label_range_real,
                                          label_range_fake=config.label_range_fake)
        self.criterionPix = config.criterion_pix
        self.criterionEmotion = config.criterion_emotion
        self.criterionVGG = utils.VGGLoss(self.device)

        # Set directories
        self.save_dir = 'saves/pix2pix/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.img_dir = os.path.join(self.save_dir, 'images')

        # Init tensorboard
        if self.config.log_run:
            os.makedirs(self.img_dir, exist_ok=True)
            self.writer = SummaryWriter(self.save_dir)
            wandb.init(project="emotion-pix2pix", config=config, sync_tensorboard=True)
            wandb.watch(self.generator)
            wandb.watch(self.discriminator)
        else:
            self.writer = None

        # Init variables
        self.loss_G_GAN = torch.tensor(0.)
        self.loss_G_pixel = torch.tensor(0.)
        self.loss_G_emotion = torch.tensor(0.)
        self.loss_G_VGG = torch.tensor(0.)
        self.loss_G_total = torch.tensor(0.)
        self.loss_D_real = torch.tensor(0.)
        self.loss_D_fake = torch.tensor(0.)
        self.loss_D_total = torch.tensor(0.)

        self.epoch_loss_G_GAN = torch.tensor(0.)
        self.epoch_loss_G_pixel = torch.tensor(0.)
        self.epoch_loss_G_emotion = torch.tensor(0.)
        self.epoch_loss_G_VGG = torch.tensor(0.)
        self.epoch_loss_G_total = torch.tensor(0.)
        self.epoch_loss_D_real = torch.tensor(0.)
        self.epoch_loss_D_fake = torch.tensor(0.)
        self.epoch_loss_D_total = torch.tensor(0.)

        self.epoch_metric_names = [
            'epoch_loss_D_total',
            'epoch_loss_D_fake',
            'epoch_loss_D_real',
            'epoch_loss_G_emotion',
            'epoch_loss_G_pixel',
            'epoch_loss_G_VGG',
            'epoch_loss_G_GAN',
        ]

    def save(self):
        """
        Save models

        args:
            epoch (int): Current epoch
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s.pt' % (name)
                save_path = os.path.join(self.save_dir, save_filename)
                print('Saving {} to {}'.format(name, save_path))

                os.makedirs(self.save_dir, exist_ok=True)
                model = getattr(self, name)
                torch.save(model.state_dict(), save_path)

                if self.config.log_run:
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, save_filename))

        print()

    def _make_grid_image(self, real_A, real_B, fake_B):
        real_A = real_A[0]
        real_B = real_B[0]
        fake_B = fake_B[0]

        # Denormalize one sequence
        if self.config.normalize:
            transform = utils.denormalize(self.config.mean, self.config.std)
            real_A = torch.stack([transform(a) for a in real_A], 0).detach()
            real_B = torch.stack([transform(a) for a in real_B], 0).detach()
            fake_B = torch.stack([transform(a) for a in fake_B], 0).detach()

        # Make grid image
        grid_image = torch.cat((real_A, fake_B, real_B), -2)
        grid_image = make_grid(grid_image, nrow=real_A.size(0), normalize=False)

        return grid_image

    def sample_images(self, data_loaders):
        """
        Saves a generated sample
        """
        # Generate training sample
        train_batch = next(iter(data_loaders['train']))
        self.set_inputs(train_batch)
        with torch.no_grad():
            fake_B = self.generator(self.real_A[0].unsqueeze(0), self.cond[0].unsqueeze(0))
        grid_image_train = self._make_grid_image(self.real_A, self.real_B, fake_B).cpu()
        train_label = self._map_labels(self.cond[0])

        # Generate validation sample
        val_batch = next(iter(data_loaders['val']))
        self.set_inputs(val_batch)
        with torch.no_grad():
            fake_B = self.generator(self.real_A[0].unsqueeze(0), self.cond[0].unsqueeze(0))
        grid_image_val = self._make_grid_image(self.real_A, self.real_B, fake_B).cpu()
        val_label = self._map_labels(self.cond[0])

        # Create figure
        f = plt.figure(figsize=(6, 6))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Train Images, {}".format(train_label))
        plt.imshow(np.transpose(grid_image_train, (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Val Images, {}".format(val_label))
        plt.imshow(np.transpose(grid_image_val, (1, 2, 0)))

        # Convert figure to array
        img_sample = utils.fig2img(f)

        return img_sample

    def _zero_running_metrics(self):
        for name in self.epoch_metric_names:
            if isinstance(name, str):
                setattr(self, name, torch.tensor(0.))

    def _mean_running_metrics(self, len_loader):
        for name in self.epoch_metric_names:
            if isinstance(name, str):
                metric = getattr(self, name)
                setattr(self, name, metric / len_loader)

    def _map_labels(self, label):

        # One-hot encoding
        if len(label.shape) == 2:
            label = torch.where(label == 1.)[1]
        elif len(label.shape) == 1 and label.sum() == 1.:
            label = torch.where(label == 1.)

        return self.mapping_list[label]

    def log_tensorboard(self):
        """
        Log metrics to tensorboard
        """
        self.writer.add_scalar('G/loss/GAN', self.loss_G_GAN, self.global_step)
        self.writer.add_scalar('G/loss/pixel', self.loss_G_pixel, self.global_step)
        self.writer.add_scalar('G/loss/emotion', self.loss_G_emotion, self.global_step)
        self.writer.add_scalar('D/loss/real', self.loss_D_real, self.global_step)
        self.writer.add_scalar('D/loss/fake', self.loss_D_fake, self.global_step)
        self.writer.add_scalar('D/loss/total', self.loss_D_total, self.global_step)

    def log_console(self, i_epoch):
        print("G loss GAN: {:.3f}\tG loss Pix: {:.3f}\tG loss Emo: {:.3f}".format(
            self.epoch_loss_G_GAN, self.epoch_loss_G_pixel, self.epoch_loss_G_emotion
        ))
        print("D loss total: {:.3f}\tD loss real: {:.3f}\tD loss fake: {:.3f}".format(
            self.epoch_loss_D_total, self.epoch_loss_D_real, self.epoch_loss_D_fake
        ))
        print('Time elapsed {} | Time left: {}\n'.format(
            utils.time_to_str(time.time() - self.t_start),
            utils.time_left(self.t_start, self.config.num_epochs, i_epoch)
        ))

    def set_inputs(self, inputs):
        """
        Unpack input data from the dataloader

        args:
            inputs (dict): Packaged input data
        """
        self.real_A = inputs['A'].to(self.device)
        self.real_B = inputs['B'].to(self.device)

        if 'y' in inputs.keys():
            self.cond = inputs['y'].to(self.device)
        else:
            self.cond = torch.tensor(0.).to(self.device)

    def forward(self):
        """
        Run forward pass
        """
        self.fake_B = self.generator(self.real_A, self.cond)

    def backward_D(self):
        """
        Compute losses for the discriminator
        """
        # All real batch
        pred_real = self.discriminator(
            {'img_a': self.real_A, 'img_b': self.real_B, 'cond': self.cond})
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        # All fake batch
        pred_fake = self.discriminator(
            {'img_a': self.real_A, 'img_b': self.fake_B.detach(), 'cond': self.cond})
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)

        # Combine losses
        self.loss_D_total = self.loss_D_fake + self.loss_D_real

        # Metrics
        self.epoch_loss_D_real += self.loss_D_real.item()
        self.epoch_loss_D_fake += self.loss_D_fake.item()
        self.epoch_loss_D_total += self.loss_D_total.item()

        # Backward
        self.loss_D_real.backward()
        self.loss_D_fake.backward()

    def backward_G(self):
        """
        Compute losses for the generator
        """
        # GAN loss
        if self.config.lambda_G_GAN:
            pred_fake = self.discriminator(
                {'img_a': self.real_A, 'img_b': self.fake_B, 'cond': self.cond})
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)
        else:
            self.loss_G_GAN = torch.tensor(0.)

        # Pixelwise loss
        if self.config.lambda_pixel:
            self.loss_G_pixel = self.criterionPix(self.fake_B, self.real_B)
        else:
            self.loss_G_pixel = torch.tensor(0.)

        # Emotion loss
        if self.config.lambda_emotion:
            embedding_fake = self.classifier(self.fake_B)
            embedding_real = self.classifier(self.real_B)
            self.loss_G_emotion = self.criterionEmotion(embedding_fake, embedding_real)
        else:
            self.loss_G_emotion = torch.tensor(0.)

        # VGG loss
        if self.config.lambda_vgg:
            self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B)
        else:
            self.loss_G_VGG = torch.tensor(0.)

        # Combine losses
        self.loss_G_total = self.loss_G_GAN * self.config.lambda_G_GAN \
                            + self.loss_G_pixel * self.config.lambda_pixel \
                            + self.loss_G_VGG * self.config.lambda_vgg \
                            + self.loss_G_emotion * self.config.lambda_emotion

        # Metrics
        self.epoch_loss_G_GAN += self.loss_G_GAN.item()
        self.epoch_loss_G_total += self.loss_G_total.item()
        self.epoch_loss_G_pixel += self.loss_G_pixel.item()
        self.epoch_loss_G_VGG += self.loss_G_VGG.item()
        self.epoch_loss_G_emotion += self.loss_G_emotion.item()

        # Backward
        self.loss_G_total.backward()

    def train_model(self, data_loaders, plot_grads=False):

        print("Starting training")
        self.t_start = time.time()

        for i_epoch in range(1, self.config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, self.config.num_epochs))
            print('-' * 10)

            self._zero_running_metrics()

            for batch in data_loaders['train']:

                # Increment step counter
                self.global_step += 1

                # Set inputs
                self.set_inputs(batch)

                # Forward
                self.forward()

                # (1) Train discriminator
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                # (2) Train Generator
                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()

                # Tensorboard logging
                if self.config.log_run:
                    self.log_tensorboard()

            # Epoch finished
            self._mean_running_metrics(len(data_loaders['train']))

            self.log_console(i_epoch)

            if self.config.log_run and i_epoch % self.config.save_interval == 0:
                # Generate sample images
                img_sample = self.sample_images(data_loaders)

                # Save image (Important: wirter.add_image has to be before save_image!!!)
                self.writer.add_image('sample', img_sample, i_epoch)
                save_image(img_sample, self.img_dir + '/sample_{}.png'.format(i_epoch))

                # Save model
                self.save()

        # Finished training
        print("Training finished in {}".format(utils.time_to_str(time.time() - self.t_start)))

        if self.config.log_run:
            self.save()
            self.eval_model(data_loaders)

    def eval_model(self, data_loaders):
        # Real images vs fake images
        batch = next(iter(data_loaders['val']))
        self.set_inputs(batch)
        with torch.no_grad():
            fake_B = self.generator(self.real_A, self.cond)

        real_B = self.real_B[:, 0]
        fake_B = fake_B[:, 0]

        # Denormalize
        if self.config.normalize:
            transform = utils.denormalize(self.config.mean, self.config.std)
            real_B = torch.stack([transform(a) for a in real_B], 0).detach()
            fake_B = torch.stack([transform(a) for a in fake_B], 0).detach()

        real_img = make_grid(real_B, padding=5, normalize=False)
        fake_img = make_grid(fake_B, padding=5, normalize=False)

        real_img = torch.nn.functional.pad(real_img, [0, 30, 0, 0], mode='constant')

        # Cat real and fake together
        imgs = torch.cat((real_img, fake_img), -1)
        imgs = make_grid(imgs, nrow=1, normalize=False)

        self.writer.add_image('random_samples', imgs)
        save_image(imgs, os.path.join(self.img_dir, 'random_samples.png'))
