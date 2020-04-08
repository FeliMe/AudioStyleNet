import torch
import torch.nn as nn
import torch.nn.functional as F

from dreiDDFA.model import dreiDDFA


class LandmarksLoss(nn.Module):
    def __init__(self, dense, mean=0., std=1.):
        super(LandmarksLoss, self).__init__()

        self.model = dreiDDFA()
        self.dense = dense
        self.mean = mean
        self.std = std

    def prepare_input(self, input_img):
        img = (input_img * self.std) + self.mean
        param_dict = self.model.predict_param(img)
        return param_dict

    def get_loss(self, input_param, input_roi_boxes, target_param, target_roi_boxes):
        input_lm = self.model.reconstruct_vertices(
            input_param, input_roi_boxes, dense=self.dense)

        target_lm = self.model.reconstruct_vertices(
            target_param, target_roi_boxes, dense=self.dense)

        loss = F.mse_loss(input_lm, target_lm)

        return loss

    def forward(self, input_img, target_param):
        input_param = self.prepare_input(input_img)
        if input_param is None:
            print("LandmarksLoss failed to find face, returns 0 loss")
            loss = torch.tensor(0.)
        else:
            loss = self.get_loss(input_param['param'], input_param['roi_box'],
                                 target_param['param'], target_param['roi_box'])

        return loss
