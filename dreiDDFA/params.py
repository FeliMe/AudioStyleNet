import os.path as osp
import numpy as np
import pickle
import torch


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


def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


d = make_abs_path('config')
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))
meta = _load(osp.join(d, 'param_whitening.pkl'))

# param_mean and param_std are used for re-whitening
param_mean = torch.tensor(meta.get('param_mean')).unsqueeze(0)
param_std = torch.tensor(meta.get('param_std')).unsqueeze(0)
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp

# for inference
u_base = u[:, keypoints].view(1, -1, 1)
w_shp_base = w_shp[:, keypoints]
w_exp_base = w_exp[:, keypoints]
std_size = 120
