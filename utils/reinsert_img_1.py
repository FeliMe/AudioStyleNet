import cv2
import dlib
import numpy as np

from PIL import Image
from skimage import io


def show_landmarks(landmarks, image_size):
    if not type(landmarks) == list:
        landmarks = [landmarks]
    img = np.zeros(image_size, dtype=np.uint8)
    for lm in landmarks:
        x, y, w, h = cv2.boundingRect(lm)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        for (x, y) in lm:
            img = cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
    cv2.imshow("", img)
    cv2.waitKey(0)


def show_image(img):
    Image.fromarray(img).show()


class FaceInsertion:
    def __init__(self):
        # Init face tracking
        predictor_path = '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat'
        self.landmark_detector = dlib.shape_predictor(predictor_path)
        detector_path = '/home/meissen/Datasets/mmod_human_face_detector.dat'
        self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)

    def get_landmarks(self, img):
        rects = self.face_detector(img, 1)
        if len(rects) == 0:
            return None
        rect = rects[0].rect
        pts = self.landmark_detector(img, rect).parts()
        pts = np.array([(pt.x, pt.y) for pt in pts], dtype=np.int32)
        return pts

    @staticmethod
    def clean_edges(img, pixels_to_ommit=2):
        """
        This function removes the first #pixels_to_ommit pixels that are non black from every side of the image
        (top, bottom, left, right)
        args:
            img (np.array): cv2 image
            pixels_to_ommit (int): how many pixels to remove
        """
        w, h, _ = img.shape

        for _k, _j in [[[0, h, 1], [0, w, 1]], [[h - 1, -1, -1], [w - 1, -1, -1]]]:
            for k in range(_k[0], _k[1], _k[2]):
                passed_non_black_pixels = 0
                for j in range(_j[0], _j[1], _j[2]):
                    if not (img[j, k, :] == [0, 0, 0]).all():
                        passed_non_black_pixels += 1
                        if passed_non_black_pixels > pixels_to_ommit:
                            break
                        else:
                            img[j, k, :] = [0, 0, 0]

            for j in range(_j[0], _j[1], _j[2]):
                passed_non_black_pixels = 0
                for k in range(_k[0], _k[1], _k[2]):
                    if not (img[j, k, :] == [0, 0, 0]).all():
                        passed_non_black_pixels += 1
                        if passed_non_black_pixels > pixels_to_ommit:
                            break
                        else:
                            img[j, k, :] = [0, 0, 0]

        return img

    @staticmethod
    def get_mask(img, landmarks):
        # Create empty mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Fill mask with convex hull of langmarks
        hull = cv2.convexHull(landmarks)
        mask = cv2.fillConvexPoly(mask, hull, 255)

        return mask

    @staticmethod
    def smooth_mask(mask, smooth):
        # as a gaussian blur can only make my region of interest smaller we need to dilate mask to find balance
        k_size = 2 * smooth + 1
        k_dilate = k_size // 4
        kernel = np.ones((k_dilate, k_dilate), np.uint8)

        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)

        return mask

    def get_masked_img(self, img, landmarks):
        # Get mask
        mask = self.get_mask(img, landmarks)

        # Apply mask to image
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # Crop masked image to contain only rect
        x, y, w, h = cv2.boundingRect(landmarks)
        masked_img = masked_img[y:y + h, x:x + w]
        mask = mask[y:y + h, x:x + w]

        return masked_img, mask

    @staticmethod
    def reinsert_image(source, target, start_pt):
        target_img = target.copy()
        y_size, x_size = source.shape[:2]
        for k in range(y_size):
            for j in range(x_size):
                y = start_pt[1] + k
                x = start_pt[0] + j
                if not (source[k, j, :] == [0, 0, 0]).all():
                    try:
                        target_img[y, x, :] = source[k, j, :]
                    except IndexError:
                        # this happens at the edges of the image and can safely be ignored
                        pass
        return target_img

    def insert_face(self, img_ori, img_mod, save_dir):
        # Get landmarks
        lm_mod = self.get_landmarks(img_mod)
        if lm_mod is None:
            print("Failed to find face in modified image, returning None")
            return None

        lm_ori = self.get_landmarks(img_ori)
        if lm_ori is None:
            print("Failed to find face in original image, returning None")
            return None

        # Get only face of modified image
        masked_img_mod, mask_mod = self.get_masked_img(img_mod, lm_mod)

        # Get bounding boxes of landmarks
        x_ori, y_ori, w_ori, h_ori = cv2.boundingRect(lm_ori)
        x_mod, y_mod, w_mod, h_mod = cv2.boundingRect(lm_mod)

        # Move landmarks to upper left of respective image
        lm_mod_norm = lm_mod.copy()
        lm_mod_norm[:, 0] -= x_mod
        lm_mod_norm[:, 1] -= y_mod
        lm_ori_norm = lm_ori.copy()
        lm_ori_norm[:, 0] -= x_ori
        lm_ori_norm[:, 1] -= y_ori

        # Select three points from the normalised landmarks to compute the affine parameters
        points_ori = np.stack(
            (lm_ori_norm[0], lm_ori_norm[33], lm_ori_norm[16])).astype(np.float32)
        points_mod = np.stack(
            (lm_mod_norm[0], lm_mod_norm[33], lm_mod_norm[16])).astype(np.float32)

        # Get transformation matrix
        M = cv2.getAffineTransform(points_mod, points_ori)

        # Warp modified image to original image shape and size
        warped_img_mod = cv2.warpAffine(masked_img_mod, M, (w_ori, h_ori))

        # Remove empty edges
        warped_img_mod = self.clean_edges(warped_img_mod, pixels_to_ommit=3)

        final = self.reinsert_image(warped_img_mod, img_ori, (x_ori, y_ori))
        show_image(final)


if __name__ == '__main__':

    original = 'saves/test/01-01-01-01-01-01-01_00001.png'
    modified = 'saves/test/01-01-01-01-01-01-01_happy.png'

    img_ori = io.imread(original)
    img_mod = io.imread(modified)

    swap = FaceInsertion()
    swap.insert_face(img_ori, img_mod, 'saves/test/temp/')
