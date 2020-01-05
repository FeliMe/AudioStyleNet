"""
File for general usefull functions which are not specific to a certain module
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def denormalize(mean, std):
    """ Denormalizes image to save or display it """
    return transforms.Compose([
        transforms.Normalize([0., 0., 0.], 1 / np.array(std)),
        transforms.Normalize(-np.array(mean), [1., 1., 1.])]
    )


class GradPlotter:
    """
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    Plots gradients flowing through different layers in the net during training.
    Can be used for checking possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    'if grad_plotter is None:
        grad_plotter = GradPlotter(self.model.named_parameters())
    grad_plotter.plot_grad_flow(self.model.named_parameters())'
    to visualize the gradient flow
    """
    def __init__(self, named_parameters):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                # if n != 'fc.weight':
                #     print(n, torch.sum(p.grad.abs()))
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        bar1 = ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        bar2 = ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")

        ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        ax.set_xticks(range(0, len(ave_grads), 1), layers)
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001,
                    top=max([tensor.cpu() for tensor in ave_grads]))  # zoom in on the lower gradient regions
        ax.set_xlabel("Layers")
        ax.set_ylabel("average gradient")
        ax.set_title("Gradient flow")
        ax.grid(True)
        ax.legend([plt.Line2D([0], [0], color="c", lw=4),
                   plt.Line2D([0], [0], color="b", lw=4),
                   plt.Line2D([0], [0], color="k", lw=4)],
                  ['max-gradient', 'mean-gradient', 'zero-gradient'])

        self.fig = fig
        self.ax = ax
        self.bar1 = bar1
        self.bar2 = bar2

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

        for rect, h in zip(self.bar1, max_grads):
            rect.set_height(h)
        for rect, h in zip(self.bar2, ave_grads):
            rect.set_height(h)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def time_to_str(t):
    return "{:.0f}h {:.0f}m {:.0f}s".format(t // 3600, (t // 60) % 60, t % 60)


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (time.time() - t_start) / i_iter
    time_remaining = time_per_iter * iters_left
    return time_to_str(time_remaining)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fig2img(fig):
    """
    Convert a matplotlib figure into a PIL Image

    Arguments:
        fig (matplotlib.figure.Figure): Input figure

    Returns:
        img (PIL.Image.Image): Output image
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(w, h, 4)
    buf = np.roll(buf, 3, axis=2).transpose((2, 0, 1))
    img = torch.tensor(buf / 255.)
    return img


class GANLoss(nn.Module):
    # Source: https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py
    """
    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, device, target_real_label=1.0,
                 target_fake_label=0.0, flip_p=0., noisy_labels=False,
                 label_range_real=(1.0, 1.0), label_range_fake=(0.0, 0.0)):
        """
        Initialize the GANLoss class.

        Parameters:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgan.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image
            flip_p (float): probability of flipping labels in discriminator
            noisy_labels (bool): Use noisy labels or not
            label_range_real (tuple of floats): Min and max for real labels if noisy_labels == True
            label_range_fake (tuple of floats): Min and max for fake labels if noisy_labels == True

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.device = device

        # Training labels
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # Flip labels
        self.flip_prob = torch.distributions.bernoulli.Bernoulli(flip_p)

        # Noisy labels
        self.noisy_labels = noisy_labels
        self.label_range_real = label_range_real
        self.label_range_fake = label_range_fake

        # Gan mode
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            pass
        elif gan_mode == 'vanilla':
            pass
        elif gan_mode == 'wgan':
            pass
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real, for_discriminator):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor): tpyically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
            if for_discriminator:
                if self.noisy_labels:
                    mini, maxi = self.label_range_real
                    noise = torch.rand(prediction.size()) * (maxi - mini) - (1. - abs(mini))
                    target_tensor = target_tensor + noise
                target_tensor = self.flip_labels(target_tensor)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
            if for_discriminator:
                if self.noisy_labels:
                    mini, maxi = self.label_range_fake
                    noise = torch.rand(prediction.size()) * (maxi - mini) - (0. - abs(mini))
                    target_tensor = target_tensor + noise
                target_tensor = self.flip_labels(target_tensor)

        return target_tensor.to(self.device)

    def flip_labels(self, target_tensor):
        """
        Randomly flip labels of target_tensor with a probability provided by
        flip_p during init.

        Parameters:
            target_tensor (torch.tensor): tensor with labels
        """
        flip_idx = self.flip_prob.sample((target_tensor.size())).bool()
        target_tensor[flip_idx] = 1 - target_tensor[flip_idx]
        return target_tensor

    def loss(self, prediction, target_is_real, for_discriminator=True):
        if self.gan_mode == 'vanilla':  # cross entropy loss
            target_tensor = self.get_target_tensor(prediction, target_is_real, for_discriminator)
            loss = F.binary_cross_entropy_with_logits(prediction, target_tensor)
            return loss
        elif self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(prediction, target_is_real, for_discriminator)
            return F.mse_loss(prediction, target_tensor)
        else:
            # wgan
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """
        Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor): tpyically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images
            for_discriminator (bool): Indicates discriminator. Noisy labels are only for D
        Returns:
            the calculated loss.
        """
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(prediction, list):
            loss = 0
            for pred_i in prediction:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(prediction)
        else:
            return self.loss(prediction, target_is_real, for_discriminator)
