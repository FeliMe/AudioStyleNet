import meshio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreiDDFA.model import dreiDDFA


class LandmarksLoss(nn.Module):
    def __init__(self, dense, img_mean=0., img_std=1.):
        super(LandmarksLoss, self).__init__()

        self.model = dreiDDFA(dense=dense)
        self.dense = dense
        self.img_mean = img_mean
        self.img_std = img_std

        self.register_buffer('mask', self.load_mask())
        self.loss_fn = F.mse_loss

    def prepare_input(self, input_img):
        img = ((input_img * self.img_std) + self.img_mean).clamp(-1., 1.)
        param_dict = self.model.predict_param(img)
        return param_dict

    def load_mask(self):
        if self.dense:
            red = meshio.read('saves/pre-trained/mouth_mask2_3ddfa.ply').point_data['red']
            return torch.tensor(np.where(red == -4, 1., 0.), dtype=torch.float32).view(1, -1, 1)
        else:
            mask = torch.zeros((1, 68, 3), dtype=torch.float32)
            mask[:, 48:68] = 1.
            return mask

    def get_vertices(self, input_param, input_roi_boxes, target_param, target_roi_boxes):
        input_lm = self.model.reconstruct_vertices(
            input_param, input_roi_boxes, dense=self.dense)

        target_lm = self.model.reconstruct_vertices(
            target_param, target_roi_boxes, dense=self.dense)

        return input_lm, target_lm

    def correct_pose(self, lm, param):
        R, t = self.model.parse_pose(param)
        lm = torch.bmm(lm, R)
        # Translation is usually wrong because of depth
        lm = lm - t
        # 8207 is a point below the noise to correct depth
        lm = lm - lm[:, 8207].unsqueeze(1)
        return lm

    def forward(self, inp, target):
        input_param = inp if type(inp) is dict else self.prepare_input(inp)
        target_param = target if type(target) is dict else self.prepare_input(target)

        # Check failed
        if input_param is None:
            print("LandmarksLoss failed to find face, returns 0 loss")
            return torch.tensor(0.)

        # Get vertices of input and target
        input_lm, target_lm = self.get_vertices(
            input_param['param'], input_param['roi_box'], target_param['param'], target_param['roi_box'])

        # Get pose
        input_lm = self.correct_pose(input_lm, input_param['param'])
        target_lm = self.correct_pose(target_lm, target_param['param'])
        input_lm *= self.mask
        target_lm *= self.mask

        # Visualize
        # from dreiDDFA.ddfa_utils import plot_pointclouds
        # plot_pointclouds([input_lm[0].detach().cpu(), target_lm[0].detach().cpu()])
        # 1 / 0

        # Get actual loss
        loss = self.loss_fn(input_lm, target_lm, reduction='none')
        loss = (loss * self.mask).sum() / self.mask.sum()

        return loss
