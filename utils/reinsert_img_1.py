import cv2
import dlib
import numpy as np

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

    def get_mask(self, img, landmarks):
        # Create empty mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Fill mask with convex hull of langmarks
        hull = cv2.convexHull(landmarks)
        mask = cv2.fillConvexPoly(mask, hull, 255)

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

    def insert_face(self, original, modified, save_dir):
        # Load images
        img_ori = io.imread(original)
        img_mod = io.imread(modified)

        # Get landmarks
        lm_mod = self.get_landmarks(img_mod)
        if lm_mod is None:
            print(f"Failed to find face in {modified}, returning None")
            return None

        lm_ori = self.get_landmarks(img_ori)
        if lm_ori is None:
            print(f"Failed to find face in {original}, returning None")
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
        warped_mask_mod = cv2.warpAffine(mask_mod, M, (w_ori, h_ori))
        alpha_s = warped_mask_mod / 255.
        alpha_l = 1.0 - alpha_s

        final = img_ori.copy()
        for c in range(0, 3):
            final[y_ori:y_ori + h_ori, x_ori:x_ori + w_ori, c] = (alpha_s * warped_img_mod[:, :, c] +
                                                                  alpha_l * final[y_ori:y_ori + h_ori, x_ori:x_ori + w_ori, c])
        cv2.imshow("", final)
        cv2.waitKey(0)


if __name__ == '__main__':

    original = 'saves/test/01-01-01-01-01-01-01_00001.png'
    modified = 'saves/test/01-01-01-01-01-01-01_happy.png'

    swap = FaceInsertion()
    swap.insert_face(original, modified, 'saves/test/temp/')
