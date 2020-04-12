import dreiDDFA.ddfa_utils as d_utils
import dlib
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreiDDFA.mobilenet_v1 import pretrained_model
from math import sqrt
from os import path as osp


class dreiDDFA(nn.Module):
    def __init__(self, dense):
        super(dreiDDFA, self).__init__()

        self.dense = dense

        # 1. load pre-tained model
        self.model = pretrained_model()
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. load dlib model for face detection and landmark used for face cropping
        regressor_path = '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat'
        self.face_regressor = dlib.shape_predictor(regressor_path)
        detector_path = '/home/meissen/Datasets/mmod_human_face_detector.dat'
        self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)
        # self.face_detector = dlib.get_frontal_face_detector()

        # 3. Init params
        self.init_params()

        # Select plot_fn
        self.plot_fn = d_utils.plot_vertices if dense else d_utils.draw_landmarks

    def init_params(self):
        def _get_suffix(filename):
            """a.jpg -> jpg"""
            pos = filename.rfind('.')
            if pos == -1:
                return ''
            return filename[pos + 1:]

        def _load(fp):
            suffix = _get_suffix(fp)
            if suffix == 'npy':
                return torch.tensor(np.load(fp)).unsqueeze(0)
                # return np.load(fp)
            elif suffix == 'pkl':
                return pickle.load(open(fp, 'rb'))

        d = osp.join(osp.dirname(osp.realpath(__file__)), 'config')
        self.register_buffer('keypoints', _load(osp.join(d, 'keypoints_sim.npy')))
        self.register_buffer('w_shp', _load(osp.join(d, 'w_shp_sim.npy')))
        self.register_buffer('w_exp', _load(osp.join(d, 'w_exp_sim.npy')))

        # param_mean and param_std are used for re-whitening
        meta = _load(osp.join(d, 'param_whitening.pkl'))
        self.register_buffer('param_mean', torch.tensor(meta.get('param_mean')).unsqueeze(0))
        self.register_buffer('param_std', torch.tensor(meta.get('param_std')).unsqueeze(0))
        self.register_buffer('u_shp', _load(osp.join(d, 'u_shp.npy')))
        self.register_buffer('u_exp', _load(osp.join(d, 'u_exp.npy')))
        self.register_buffer('u', self.u_shp + self.u_exp)

        # for inference
        self.register_buffer('u_base', self.u[:, self.keypoints].view(1, -1, 1))
        self.register_buffer('w_shp_base', self.w_shp[:, self.keypoints])
        self.register_buffer('w_exp_base', self.w_exp[:, self.keypoints])
        self.std_size = 120

    def get_roi(self, img_torch):
        img_np = (img_torch.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        roi_boxes = []
        for img in img_np:
            # Detect face
            rects = self.face_detector(img, 1)
            if len(rects) == 0:
                return None
            rect = rects[0].rect

            # Get region of interest
            pts = self.face_regressor(img, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = d_utils.parse_roi_box_from_landmark(pts)
            roi_boxes.append(roi_box)

        return roi_boxes

    def reconstruct_vertices(self, param, roi_bbox, dense):
        vertex = self.reconstruct_vertex(param, dense=dense)
        for i in range(len(roi_bbox)):
            sx, sy, ex, ey = roi_bbox[i]
            scale_x = (ex - sx) / 120
            scale_y = (ey - sy) / 120
            vertex[i, :, 0] = vertex[i, :, 0] * scale_x + sx
            vertex[i, :, 1] = vertex[i, :, 1] * scale_y + sy

            s = (scale_x + scale_y) / 2
            vertex[i, :, 2] *= s

        return vertex

    def reconstruct_vertex(self, param, whitening=True, dense=False, transform=True):
        """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
        image coordinate space, but without alignment caused by face cropping.
        transform: whether transform to image space
        """
        b = param.shape[0]
        if param.shape[-1] == 12:
            param = torch.cat((param, [0] * 50))
        if whitening:
            param = param * self.param_std + self.param_mean

        p, offset, alpha_shp, alpha_exp = d_utils._parse_param(param)

        if dense:
            vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @
                          alpha_exp).view(b, -1, 3).transpose(2, 1) + offset

            if transform:
                # transform to image coordinate space
                vertex[:, 1, :] = self.std_size + 1 - vertex[:, 1, :]
        else:
            """For 68 pts"""
            vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @
                          alpha_exp).view(b, -1, 3).transpose(2, 1) + offset

            if transform:
                # transform to image coordinate space
                vertex[:, 1, :] = self.std_size + 1 - vertex[:, 1, :]

        return vertex.transpose(2, 1)

    def parse_pose(self, param):
        b = param.shape[0]
        param = param * self.param_std + self.param_mean
        Ps = param[:, :12].view(b, 3, -1)  # camera matrix
        s, R, t3d = d_utils.P2sRt(Ps)
        # P = torch.cat((R, t3d.view(b, 3, -1)), dim=2)  # without scale
        # P = Ps / s
        # pose = d_utils.matrix2angle(R)  # yaw, pitch, roll
        return R.transpose(2, 1), t3d.view(b, 1, 3)

    def predict_param(self, image):
        assert len(image.shape) == 4, "3DDFA model accepts only input of shape NCHW"

        # Get region of interest
        roi_boxes = self.get_roi(image)
        if roi_boxes is None:
            return None

        # Crop images
        img_crop = [d_utils.crop_img(img, box) for img, box in zip(image, roi_boxes)]

        # Resize images
        img_crop = torch.cat([F.interpolate(img.unsqueeze(
            0), size=self.std_size, mode='nearest') for img in img_crop])

        # Rescale and convert from RGB to BGR
        img_crop = ((img_crop * 2) - 1)[:, [2, 1, 0]]

        # Forward
        param = self.model(img_crop)

        return {'param': param, 'roi_box': roi_boxes}

    def forward(self, image):
        param_dict = self.predict_param(image)

        # Forward
        vertices = self.reconstruct_vertices(
            param_dict['param'], param_dict['roi_box'], dense=self.dense)

        param_dict['vertices'] = vertices

        # Visualize
        # self.plot_fn(image[0].detach().cpu(), vertices[0].detach().cpu())
        # 1 / 0

        return param_dict
