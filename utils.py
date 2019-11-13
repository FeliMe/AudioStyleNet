"""
File for general usefull functions which are not specific to a certain module
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.transforms as transforms


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def denormalize(mean, std):
    """ Denormalizes image to save or display it """
    return transforms.Compose([
        transforms.Normalize([0., 0., 0.], 1/np.array(std)),
        transforms.Normalize(-np.array(mean), [1., 1., 1.])]
    )


class GradPlotter:
    """
    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

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
    return "{:.0f}h {:.0f}m {:.0f}s".format(t // 3600, t // 60, t % 60)


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (time.time() - t_start) / i_iter
    time_remaining = time_per_iter * iters_left
    return time_to_str(time_remaining)


def ada_loss(curr_loss, prev_v, beta=0.99):
    curr_loss = curr_loss.view((curr_loss.size(0), -1)).mean(1)
    curr_v = beta * prev_v + (1 - beta) * curr_loss.pow(2).mean()
    curr_v_ = curr_v / (1 - beta)

    loss = curr_loss.mean() / (curr_v_.sqrt() + 1e-10)

    return loss, curr_v


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
