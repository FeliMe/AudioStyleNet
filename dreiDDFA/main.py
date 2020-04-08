import argparse
import dlib
import mobilenet_v1
import numpy as np
import torch
import torchvision.transforms as transforms

import ddfa_utils as d_utils
from PIL import Image

STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    model = mobilenet_v1.pretrained_model().to(args.device)

    # 2. load dlib model for face detection and landmark used for face cropping
    regressor_path = '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(regressor_path)
    detector_path = '/home/meissen/Datasets/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(detector_path)
    # face_detector = dlib.get_frontal_face_detector()

    # 3. Load image
    img_torch = transforms.ToTensor()(Image.open(args.image))
    img_np = (img_torch.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Detect face
    rects = face_detector(img_np, 1)
    if rects is None:
        print("No face detected")
        return
    rect = rects[0].rect
    # rect = rects[0]

    # Get region of interest
    pts = face_regressor(img_np, rect).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box = d_utils.parse_roi_box_from_landmark(pts)

    # Crop and resize image
    img_torch_crop = d_utils.crop_img(img_torch, roi_box)
    img_torch_crop = img_torch_crop.unsqueeze(0)
    img_torch_crop = torch.nn.functional.interpolate(
        img_torch_crop, size=STD_SIZE, mode='nearest')

    # Rescale and convert from RGB to BGR
    img_torch_crop = ((img_torch_crop * 2) - 1)[:, [2, 1, 0]]

    with torch.no_grad():
        img_torch_crop = img_torch_crop.to(args.device)
        param = model(img_torch_crop).cpu()

    vertices = d_utils.predict_vertices(param, roi_box, dense=True)
    landmarks = d_utils.predict_vertices(param, roi_box, dense=False)
    d_utils.draw_landmarks(img_torch, landmarks[0], style='fancy', show_flg=True)
    # d_utils.plot_vertices(img_torch, vertices[0])

    # P, pose = d_utils.parse_pose(param)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='../saves/test_in/00001.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # main
    main(args)
