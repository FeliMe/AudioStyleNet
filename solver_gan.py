import datetime
import os
import time
import torch
import wandb

from torchvision.utils import make_grid

import utils

from models.gan_models import weights_init


class GANSolver(object):
    def __init__(self, config):

        # General
        self.save_dir = 'saves'
        self.config = config
        self.device = 'cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu'
        self.global_step = 0
        self.t_start = 0

        print("Training on {}".format(self.device))

        # Models
        self.model_names = ['generator', 'discriminator', 'classifier']
        self.generator = config.generator.train().to(self.device)
        self.generator.apply(weights_init)

        self.discriminator = config.discriminator.train().to(self.device)
        self.discriminator.apply(weights_init)

        self.classifier = config.classifier.train().to(self.device)
        self.classifier.load_state_dict(torch.load(config.classifier_path, map_location=self.device))
        for param in self.classifier.parameters():
            param.requires_grad = False

        for name in self.model_names:
            if isinstance(name, str):
                model = getattr(self, name)
                print("{}: # params {} (trainable {})".format(
                    name,
                    utils.count_params(model),
                    utils.count_trainable_params(model)
                ))

        if config.log_run:
            wandb.watch(self.generator)
            wandb.watch(self.discriminator)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(params=self.generator.parameters(),
                                            lr=config.lr_G,
                                            betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(params=self.discriminator.parameters(),
                                            lr=config.lr_D,
                                            betas=(0.5, 0.999))

        # Loss Functions
        self.criterionGAN = utils.GANLoss(config.GAN_mode, self.device)
        self.criterionPix = config.criterion_pix
        self.criterionEmotion = config.criterion_emotion

        # Losses
        self.loss_G_GAN = torch.tensor(0.)
        self.loss_G_pixel = torch.tensor(0.)
        self.loss_G_emotion = torch.tensor(0.)
        self.loss_G_total = torch.tensor(0.)
        self.loss_D_real = torch.tensor(0.)
        self.loss_D_fake = torch.tensor(0.)
        self.loss_D_total = torch.tensor(0.)

        self.epoch_loss_G_GAN = torch.tensor(0.)
        self.epoch_loss_G_pixel = torch.tensor(0.)
        self.epoch_loss_G_emotion = torch.tensor(0.)
        self.epoch_loss_D_fake = torch.tensor(0.)
        self.epoch_loss_D_real = torch.tensor(0.)
        self.epoch_loss_D_total = torch.tensor(0.)

        # Other metrics
        self.acc_D_real = torch.tensor(0.)
        self.acc_D_fake = torch.tensor(0.)
        self.epoch_acc_D_real = torch.tensor(0.)
        self.epoch_acc_D_fake = torch.tensor(0.)

        self.epoch_acc_D_real = torch.tensor(0.)
        self.epoch_acc_D_fake = torch.tensor(0.)

        self.iteration_metric_names = ['loss_G_GAN', 'loss_G_pixel',
                                       'loss_D_total']
        self.epoch_metric_names = ['epoch_loss_G_GAN', 'epoch_loss_G_pixel',
                                   'epoch_loss_D_total', 'epoch_acc_D_real',
                                   'epoch_acc_D_fake']

    def save(self, epoch):
        """
        Save models

        args:
            epoch (int): Current epoch
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pt' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                print('Saving {} to {}'.format(name, save_path))

                os.makedirs(self.save_dir, exist_ok=True)
                model = getattr(self, name)
                torch.save(model.state_dict(), save_path)

                if self.config.log_run:
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, save_filename))

    def sample_images(self):
        """
        Saves a generated sample
        """
        # Generate fake sequence
        fake_B = self.generator(self.real_A[0].unsqueeze(0))

        # Denormalize one sequence
        transform = utils.denormalize(self.config.mean, self.config.std)
        real_a = torch.stack([transform(img) for img in self.real_A[0].data], dim=0)
        real_b = torch.stack([transform(img) for img in self.real_B[0].data], dim=0)
        fake_b = torch.stack([transform(img) for img in fake_B[0].data], dim=0)

        # Specify and create target folder
        target_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(target_dir, exist_ok=True)

        # Make grid image
        img_sample = torch.cat((real_a, fake_b, real_b), -2)
        img_sample = make_grid(img_sample, nrow=real_a.size(0), normalize=True)

        return img_sample

    def zero_running_metrics(self):
        for name in self.epoch_metric_names:
            if isinstance(name, str):
                setattr(self, name, torch.tensor(0.))

    def mean_running_metrics(self):
        for name in self.epoch_metric_names:
            if isinstance(name, str):
                metric = getattr(self, name)
                setattr(self, name, metric.mean())

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """
        Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations

        args:
            nets (network list): a list of networks
            requires_grad (bool): whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_inputs(self, inputs):
        """
        Unpack input data from the dataloader

        args:
            inputs (dict): Packaged input data
        """
        self.real_A = inputs['A'].to(self.device)
        self.real_B = inputs['B'].to(self.device)

    def forward(self):
        """
        Run forward pass
        """
        self.fake_B = self.generator(self.real_A)

    def backward_D(self):
        """
        Compute losses for the discriminator
        """
        # All real batch
        pred_real = self.discriminator(self.real_A, self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.acc_D_real = (pred_real.round() == self.criterionGAN.real_label.data).double().mean()
        self.epoch_loss_D_real += self.loss_D_real.item()
        self.epoch_acc_D_real += self.acc_D_real.item()

        # All fake batch
        pred_fake = self.discriminator(self.real_A, self.fake_B)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.acc_D_fake = (pred_fake.round() == self.criterionGAN.fake_label.data).double().mean()
        self.epoch_loss_D_fake += self.loss_D_fake.item()
        self.epoch_acc_D_fake += self.acc_D_fake.item()

        # Combined loss
        self.loss_D_total = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.epoch_loss_D_total += self.loss_D_total.item()
        self.loss_D_total.backward(retain_graph=True)

    def backward_G(self):
        """
        Compute losses for the generator
        """
        # GAN loss
        pred_fake = self.discriminator(self.real_A, self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.epoch_loss_G_GAN += self.loss_G_GAN.item()

        # Pixelwise loss
        self.loss_G_pixel = self.criterionPix(self.fake_B, self.real_B) * self.config.lambda_pixel
        self.epoch_loss_G_pixel += self.loss_G_pixel.item()

        # Emotion loss
        embedding_fake = self.classifier(self.fake_B)
        embedding_real = self.classifier(self.real_B)
        self.loss_G_emotion = self.criterionEmotion(embedding_fake, embedding_real) * self.config.lambda_emotion
        self.epoch_loss_G_emotion += self.loss_G_emotion.item()

        # Combined loss
        self.loss_G_total = self.loss_G_GAN + self.loss_G_pixel + self.loss_G_emotion
        self.loss_G_total.backward(retain_graph=False)

    def optimize_parameters(self):
        """
        Do forward and backward step and optimize parameters
        """
        self.forward()

        # Train Discriminator
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Train Generator
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def log_tensorboard(self, tb_writer):
        """
        Log metrics to tensorboard
        """
        for metric_name in self.iteration_metric_names:
            if isinstance(metric_name, str):
                # e.g. loss_G_GAN
                metric, model, name = metric_name.split('_')
                m = getattr(self, metric_name)
                tb_writer.add_scalar(model + '/' + metric + '/' + name,
                                     m, self.global_step)

    def log_console(self, i_epoch):
        for metric_name in self.epoch_metric_names:
            if isinstance(metric_name, str):
                # e.g. loss_G_GAN
                _, metric, model, name = metric_name.split('_')
                m = getattr(self, metric_name)
                m = m.mean()
                print('{} {} {}: {:.4f}'.format(model, name, metric, m))
        print('Time elapsed {}'.format(utils.time_to_str(time.time() - self.t_start)))
        print('Time left: {}\n'.format(
            utils.time_left(self.t_start, self.config.num_epochs, i_epoch)))
        max_memory = torch.cuda.max_memory_allocated(self.device) / 1e6
        print("Max memory on {}: {} MB".format(self.device, max_memory))

    def train_model(self,
                    data_loaders,
                    plot_grads=False,
                    tb_writer=None):

        self.save_dir = os.path.join(
            self.config.save_dir,
            'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        grad_plotter_g, grad_plotter_d = None, None

        print("Starting training")
        self.t_start = time.time()

        for i_epoch in range(1, self.config.num_epochs + 1):
            print('Epoch {}/{}'.format(i_epoch, self.config.num_epochs))
            print('-' * 10)

            self.zero_running_metrics()

            for batch in data_loaders['train']:

                # Increment step counter
                self.global_step += 1

                # Inputs
                self.set_inputs(batch)

                # Update parameters
                self.optimize_parameters()

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

                # Tensorboard logging
                if self.config.log_run:
                    self.log_tensorboard(tb_writer)

            # ---------------
            #  Epoch finished
            # ---------------

            self.mean_running_metrics()

            # Epoch logging
            self.log_console(i_epoch)

            if self.config.log_run:
                # Get sample from validation set
                val_batch = next(iter(data_loaders['val']))
                self.set_inputs(val_batch)

                # Generate sample images
                img_sample = self.sample_images()
                tb_writer.add_image('sample', img_sample, i_epoch)

                # Save model
                self.save(i_epoch)

        time_elapsed = time.time() - self.t_start
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        if self.config.log_run:
            self.save(-1)
