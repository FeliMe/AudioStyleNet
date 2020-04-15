import argparse
import cv2
import numpy as np
import torch


def show_landmarks(lm, image=None):
    if image is None:
        image = np.zeros((256, 256, 3))
    for (x, y) in lm:
        image = cv2.circle(image, (x, y), 1, (255, 255, 255), 1)
    cv2.imshow("", image)
    cv2.waitKey(0)


def points_to_mask(lm):
    hull = cv2.convexHull(lm.astype(np.int64))
    mask = np.zeros((256, 256, 3), dtype='uint8')
    mask = cv2.drawContours(
        mask, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask


def get_mouth_mask(landmarks, std_factor):
    mouth_lm = landmarks[:, 48:60]

    means = mouth_lm.mean(axis=0)
    std = mouth_lm.std(axis=0) * std_factor

    mask = means.copy()
    mask[0, 0] -= std[0, 0]
    mask[1] -= std[1]
    mask[2] -= std[2]
    mask[3, 1] -= std[3, 1]
    mask[4, 0] += std[4, 0]
    mask[4, 1] -= std[4, 1]
    mask[5, 0] += std[5, 0]
    mask[5, 1] -= std[5, 1]
    mask[6, 0] += std[6, 0]
    mask[7] += std[7]
    mask[8] += std[8]
    mask[9, 1] += std[9, 1]
    mask[10, 0] -= std[10, 0]
    mask[10, 1] += std[10, 1]
    mask[11, 0] -= std[11, 0]
    mask[11, 1] += std[11, 1]

    mask = points_to_mask(mask)

    # Convert to torch tensor
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    return mask


def get_eyes_mask(landmarks, std_factor):
    def get_eye_mask(mean, std):
        mask = mean.copy()
        mask[0] -= std[0]
        mask[1] -= std[1]
        mask[2, 1] -= std[2, 1]
        mask[3, 0] += std[3, 0]
        mask[3, 1] -= std[3, 1]
        mask[4, 0] += std[4, 0]
        mask[4, 1] -= std[4, 1]
        mask[5, 0] -= std[5, 0]
        mask[5, 1] += std[5, 1]
        mask[6] += std[6]
        mask[7] += std[7]
        mask[8, 0] -= std[8, 0]
        mask[8, 1] += std[8, 1]
        return mask

    # Right eye
    right_eye = np.concatenate(
        (landmarks[:, 17:22], landmarks[:, 36].reshape(-1, 1, 2), landmarks[:, 39:42]), axis=1)
    right_eye_mask = get_eye_mask(right_eye.mean(
        axis=0), right_eye.std(axis=0) * std_factor)
    right_eye_mask = points_to_mask(right_eye_mask)

    # Left eye
    left_eye = np.concatenate(
        (landmarks[:, 22:27], landmarks[:, 42].reshape(-1, 1, 2), landmarks[:, 45:48]), axis=1)
    left_eye_mask = get_eye_mask(left_eye.mean(
        axis=0), left_eye.std(axis=0) * std_factor)
    left_eye_mask = points_to_mask(left_eye_mask)

    # Combine bitwise
    mask = cv2.bitwise_or(right_eye_mask, left_eye_mask)

    # Convert to torch tensor
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    return mask


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/mnt/sdb1/meissen/Datasets/Tagesschau/Aligned256/latent_data.pt')
    parser.add_argument('--output_dir', type=str, default='../saves/pre-trained/')
    parser.add_argument('--mouth', action='store_true')
    parser.add_argument('--std_factor_mouth', type=float, default=3.)
    parser.add_argument('--eyes', action='store_true')
    parser.add_argument('--std_factor_eyes', type=float, default=3.)
    args = parser.parse_args()

    # Load data
    data = torch.load(args.data_path)
    landmarks = data['landmarks'].numpy()
    del data

    if args.mouth:
        mouth_mask = get_mouth_mask(landmarks, std_factor=args.std_factor_mouth)
        save_path = args.output_dir + f"tagesschau_mouth_mask_{int(args.std_factor_mouth)}std.pt"
        print(mouth_mask.shape, save_path)
        torch.save(mouth_mask, save_path)
    if args.eyes:
        eyes_mask = get_eyes_mask(landmarks, std_factor=args.std_factor_eyes)
        save_path = args.output_dir + f"tagesschau_eyes_mask_{int(args.std_factor_eyes)}std.pt"
        print(eyes_mask.shape, save_path)
        torch.save(eyes_mask, save_path)
