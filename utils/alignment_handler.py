import cv2
import dlib
import numpy as np
import os

RAIDROOT = os.environ['RAIDROOT']


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

    def get_landmarks(self, img):
        rects = self.face_detector(img, 1)
        if len(rects) == 0:
            return None
        rect = rects[0]
        pts = self.landmark_detector(img, rect).parts()
        pts = np.array([(pt.x, pt.y) for pt in pts], dtype=np.int32)
        return pts
