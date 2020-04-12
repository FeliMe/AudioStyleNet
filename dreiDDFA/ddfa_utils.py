import numpy as np
import torch

from math import sqrt
from dreiDDFA.params import (param_std, param_mean, u, w_shp, w_exp,
                             std_size, u_base, w_shp_base, w_exp_base)


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    return roi_box


def crop_img(img, roi_box):
    h, w = img.shape[1:]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    res = torch.zeros((3, dh, dw), dtype=torch.float32, device=img.device)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[:, dsy:dey, dsx:dex] = img[:, sy:ey, sx:ex]
    return res


def predict_vertices(param, roi_bbox, dense):
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[:, :, 0] = vertex[:, :, 0] * scale_x + sx
    vertex[:, :, 1] = vertex[:, :, 1] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[:, :, 2] *= s

    return vertex


def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    b = param.shape[0]
    if param.shape[-1] == 12:
        param = torch.cat((param, [0] * 50))
    if whitening:
        param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @
                      alpha_exp).view(b, -1, 3).transpose(2, 1) + offset

        if transform:
            # transform to image coordinate space
            vertex[:, 1, :] = std_size + 1 - vertex[:, 1, :]
    else:
        """For 68 pts"""
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @
                      alpha_exp).view(-1, 3).transpose(1, 0) + offset

        if transform:
            # transform to image coordinate space
            vertex[:, 1, :] = std_size + 1 - vertex[:, 1, :]

    return vertex.transpose(2, 1)


def _parse_param(param):
    b = param.shape[0]
    p_ = param[:, :12].view(b, 3, -1)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(b, 3, 1)
    alpha_shp = param[:, 12:52].view(b, -1, 1)
    alpha_exp = param[:, 52:].view(b, -1, 1)
    return p, offset, alpha_shp, alpha_exp


def parse_pose(param):
    b = param.shape[0]
    param = param * param_std + param_mean
    Ps = param[:, :12].view(b, 3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = torch.cat((R, t3d.view(b, 3, -1)), dim=2)  # without scale
    # P = Ps / s
    pose = matrix2angle(R)  # yaw, pitch, roll
    return P, pose


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (b, 3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (b, 3, 3). rotation matrix.
        t3d: (b, 3). 3d translation.
    '''
    t3d = P[:, :, 3]
    R1 = P[:, 0:1, :3]
    R2 = P[:, 1:2, :3]
    # pass p=2 because of bug https://github.com/pytorch/pytorch/issues/30704#issuecomment-594299501
    norm_R1 = torch.norm(R1, dim=(1, 2), keepdim=True, p=2)
    norm_R2 = torch.norm(R2, dim=(1, 2), keepdim=True, p=2)
    s = (norm_R1 + norm_R2) / 2.0
    r1 = R1 / norm_R1
    r2 = R2 / norm_R2
    r3 = torch.cross(r1, r2, dim=-1)

    R = torch.cat((r1, r2, r3), 1)
    return s, R, t3d


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (b, 3, 3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''

    if (R[:, 2, 0] == 1).any() and (R[2, 0] == -1).any():
        # Gimbal lock
        print("Gimbal lock, return ing x=0, y=0, z=0")
        return 0, 0, 0
    else:
        x = torch.asin(R[:, 2, 0])
        y = torch.atan2(R[:, 2, 1] / torch.cos(x), R[:, 2, 2] / torch.cos(x))
        z = torch.atan2(R[:, 1, 0] / torch.cos(x), R[:, 0, 0] / torch.cos(x))

    return x, y, z


def plot_pointclouds(points, title=""):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not type(points) is list:
        points = [points]
    coords = [point.unbind(1) for point in points]
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    for (x, y, z) in coords:
        ax.scatter3D(x, -z, y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(180, 90)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0, 256)
    plt.show()


def plot_vertices(image, vertices):
    import cv2
    image = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    vertices = np.round(vertices.numpy()).astype(np.int32)
    for i in range(0, vertices.shape[0], 2):
        st = vertices[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (255, 0, 0), -1)
    cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flg=True, **kwargs):
    """Draw landmarks using matplotlib"""
    import matplotlib.pyplot as plt
    # To numpy image
    img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # BGR to RGB
    img = img[:, :, [2, 1, 0]]
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if style == 'simple':
        plt.plot(pts[:, 0], pts[:, 1], 'o', markersize=4, color='g')

    elif style == 'fancy':
        alpha = 0.8
        markersize = 4
        lw = 1.5
        color = kwargs.get('color', 'w')
        markeredgecolor = kwargs.get('markeredgecolor', 'black')

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        def plot_close(i1, i2): return plt.plot([pts[i1, 0], pts[i2, 0]], [pts[i1, 1], pts[i2, 1]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[l:r, 0], pts[l:r, 1],
                     color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[l:r, 0], pts[l:r, 1], marker='o', linestyle='None', markersize=markersize,
                     color=color,
                     markeredgecolor=markeredgecolor, alpha=alpha)

    if wfp is not None:
        plt.savefig(wfp, dpi=200)
        print('Save visualization result to {}'.format(wfp))
    if show_flg:
        plt.show()
