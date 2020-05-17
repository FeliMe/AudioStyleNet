import cv2
import dlib
import numpy as np
import os
import torch
import torchgeometry as tgm

from PIL import Image
from tqdm import tqdm
from skimage import io
from torchvision import transforms


RAIDROOT = os.environ['RAIDROOT']


def torch2np_img(img):
    return (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)


def np2torch_img(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.


def show_tensor(tensor):
    if tensor.ndim == 4:
        transforms.ToPILImage()(tensor[0]).show()
    else:
        transforms.ToPILImage()(tensor).show()


class AlignmentHandler():
    def __init__(self, desiredLeftEye=(0.371, 0.470), desiredFaceShape=(256, 256), detector='frontal'):
        # Init face tracking
        predictor_path = RAIDROOT + 'Networks/shape_predictor_68_face_landmarks.dat'
        self.landmark_detector = dlib.shape_predictor(predictor_path)

        if detector == 'frontal':
            self.face_detector = dlib.get_frontal_face_detector()  # Use this one first, other for missing frames
        elif detector == 'cnn':
            detector_path = RAIDROOT + 'Networks/mmod_human_face_detector.dat'
            self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)
        else:
            raise NotImplementedError

        # Init alignment variables
        self.avg_angle = 0.
        self.initial_angle = None
        self.dist = None
        self.prev_dist = None

        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceShape = desiredFaceShape

    @staticmethod
    def align_face_static(img, keypoints, desiredLeftEye=(0.371, 0.470), desiredFaceShape=(256, 256)):
        """
        Aligns a face so that left eye is at desiredLeftEye position
        adapted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
        :param img: cv2 image, cut previously to just contain face
        :param keypoints: keypoints from dlib
        :param desiredLeftEye: position of left eye in aligned image
        :param desiredFaceShape: output image size
        :return: aligned face image, cv2
        """
        desiredFaceWidth = desiredFaceShape[0]
        desiredFaceHeight = desiredFaceShape[1]
        # get keypoints of the eyes
        leftEyePts = keypoints[36:42]
        rightEyePts = keypoints[42:48]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0)
        rightEyeCenter = rightEyePts.mean(axis=0)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))

        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) / 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        aligned_face_img = cv2.warpAffine(
            img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        return aligned_face_img, eyesCenter, angle, scale

    def reset(self):
        self.avg_angle = 0.
        self.initial_angle = None
        self.dist = None
        self.prev_dist = None

    def align_face(self, img, keypoints, desiredLeftEye=(0.371, 0.470), desiredFaceShape=(256, 256)):
        """
        Aligns a face so that left eye is at desiredLeftEye position
        adapted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
        :param img: cv2 image, cut previously to just contain face
        :param keypoints: keypoints from dlib
        :param desiredLeftEye: position of left eye in aligned image
        :param desiredFaceShape: output image size
        :return: aligned face image, cv2
        """
        desiredFaceWidth = desiredFaceShape[0]
        desiredFaceHeight = desiredFaceShape[1]
        # get keypoints of the eyes
        leftEyePts = keypoints[36:42]
        rightEyePts = keypoints[42:48]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0)
        rightEyeCenter = rightEyePts.mean(axis=0)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Smooth rotation
        if self.initial_angle is None:
            self.initial_angle = angle

        # self.avg_angle = 0.5 * self.avg_angle + \
        #     0.5 * (self.initial_angle + angle)
        self.avg_angle = 0.7 * self.avg_angle + 0.3 * angle

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))

        # Smooth dist
        if self.dist is None:
            self.prev_dist = dist

        # Reset if cut in Video (qsize makes a jump)
        if max(dist / self.prev_dist, self.prev_dist / dist) > 1.5:
            self.prev_dist = dist
            self.avg_angle = 0.
            self.initial_angle = None

        self.dist = 0.01 * dist + 0.99 * self.prev_dist
        self.prev_dist = self.dist

        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / self.dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) / 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, self.avg_angle, scale)

        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        aligned_face_img = cv2.warpAffine(
            img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        return aligned_face_img, eyesCenter, self.avg_angle, scale

    def reinsert_aligned_into_tensor(self, aligned_tensor, tensor, alignment_params, device, margin=70):
        # unpack alignmend params
        desiredLeftEye = [float(alignment_params["desiredLeftEye"][0]), float(
            alignment_params["desiredLeftEye"][1])]
        rotation_point = alignment_params["eyesCenter"]
        scale = 1 / alignment_params["scale"]
        angle = -alignment_params["angle"]

        # get original positions
        aligned_face_size = aligned_tensor.shape[-1]
        mX = round(aligned_face_size * 0.5)
        mY = round(desiredLeftEye[1] * aligned_face_size)
        # define the scale factor
        width = tensor.shape[3]
        long_edge_size = width / abs(np.cos(np.deg2rad(-angle)))
        w_original = int(scale * long_edge_size)
        h_original = int(scale * long_edge_size)
        # get offset
        tX = w_original * 0.5
        tY = h_original * desiredLeftEye[1]
        # get rotation center
        center = torch.ones(1, 2)
        center[..., 0] = mX
        center[..., 1] = mY
        # compute the transformation matrix
        M = tgm.get_rotation_matrix2d(center, angle, scale).to(device)
        M[0, 0, 2] += (tX - mX)
        M[0, 1, 2] += (tY - mY)
        # apply the transformation to original image
        _, _, h, w = aligned_tensor.shape
        aligned_tensor = tgm.warp_affine(aligned_tensor, M, dsize=(
            h_original, w_original), padding_mode='reflection')
        # get insertion point
        x_start = int(rotation_point[0] - (0.5 * w_original))
        y_start = int(rotation_point[1] - (desiredLeftEye[1] * h_original))

        if y_start < 0:
            aligned_tensor = aligned_tensor[:, :, abs(y_start):h_original, :]
            h_original += y_start
            y_start = 0
        if x_start < 0:
            aligned_tensor = aligned_tensor[:, :, :, abs(x_start):w_original]
            w_original += x_start
            x_start = 0

        _, _, h_tensor, w_tensor = tensor.shape
        if y_start + h_original > h_tensor:
            h_original -= (y_start + h_original - h_tensor)
            aligned_tensor = aligned_tensor[:, :, 0:h_original, :]
        if x_start + w_original > w_tensor:
            w_original -= (x_start + w_original - w_tensor)
            aligned_tensor = aligned_tensor[:, :, :, 0:w_original]

        # create mask
        mask = ((aligned_tensor[0][0] == 0) & (
            aligned_tensor[0][1] == 0) & (aligned_tensor[0][2] == 0))
        # remove empty edges
        aligned_tensor = torch.where(mask, tensor[:, :, y_start:y_start + h_original, x_start:x_start + w_original],
                                     aligned_tensor)

        # reinsert into tensor
        reinserted_tensor = tensor.clone()
        reinserted_tensor[0, :, y_start:y_start + h_original,
                          x_start:x_start + w_original] = aligned_tensor

        return reinserted_tensor

    def get_landmarks(self, img):
        rects = self.face_detector(img, 1)
        if len(rects) == 0:
            return None
        rect = rects[0]
        pts = self.landmark_detector(img, rect).parts()
        pts = np.array([(pt.x, pt.y) for pt in pts], dtype=np.int32)
        return pts

    def align_video(self, video, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        i_frame = 0

        self.reset()

        # Init empty alignment_params
        alignment_params = {}

        cap = cv2.VideoCapture(video)
        # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # pbar = tqdm(total=n_frames)
        while cap.isOpened():
            # Frame shape: (weight, width, 3)
            ret, frame = cap.read()
            if not ret:
                break
            i_frame += 1
            # pbar.update()
            name = str(i_frame).zfill(5) + '.png'
            save_path = os.path.join(save_dir, name)

            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pre-resize to save computation
            h_old, w_old, _ = frame.shape
            h_new = 256
            factor = h_new / h_old
            w_new = int(w_old * factor)
            frame_small = cv2.resize(frame, (w_new, h_new))

            # Get landmarks
            landmarks = self.get_landmarks(frame_small)
            if landmarks is None:
                print(f"No face found in {i_frame}, resetting aligner")
                self.reset()
                continue
            landmarks = (landmarks.astype('float') / factor).astype(np.int32)

            # Align Image
            aligned, eyesCenter, angle, scale = self.align_face(
                frame, landmarks, self.desiredLeftEye, self.desiredFaceShape)

            # Add alignment params to dict
            alignment_params[name] = {
                'desiredLeftEye': self.desiredLeftEye,
                'eyesCenter': eyesCenter,
                'scale': scale,
                'angle': angle
            }

            # Convert from numpy array to PIL Image
            aligned = Image.fromarray(aligned)

            # Visualize
            # print(save_path)
            # aligned.show()
            # 1 / 0

            # Save
            aligned.save(save_path)

        # Save alignment_params
        np.save(os.path.join(save_dir, "alignment_params.npy"), alignment_params)


if __name__ == '__main__':

    original = 'saves/test/01-01-01-01-01-01-01_00001.png'
    # original = 'saves/test/ffsmall_758_00001.png'
    swap = AlignmentHandler()

    img_ori = io.imread(original)
    lm_ori = swap.get_landmarks(img_ori)
    assert lm_ori is not None, f"Found no face in {original}"

    # Align
    desiredLeftEye = (0.371, 0.480)
    desiredFaceShape = (256, 256)
    aligned, eyesCenter, angle, scale = swap.align_face(
        img_ori, lm_ori, desiredLeftEye, desiredFaceShape)

    # Reinsert
    aligned_torch = np2torch_img(aligned).unsqueeze(0)
    ori_torch = np2torch_img(img_ori).unsqueeze(0)
    # show_tensor(aligned_torch)
    # show_tensor(ori_torch)

    alignmend_params = {
        'desiredLeftEye': desiredLeftEye,
        'eyesCenter': eyesCenter,
        'scale': torch.tensor(scale, dtype=torch.float32).unsqueeze(0),
        'angle': torch.tensor(angle, dtype=torch.float32).unsqueeze(0)
    }
    reinserted = swap.reinsert_aligned_into_tensor(
        aligned_torch, ori_torch, alignmend_params, 'cpu')
    show_tensor(reinserted)
