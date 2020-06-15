import argparse
import cv2
import face_alignment
import numpy as np
import os
import sys
import torch

from glob import glob
from eafa import Emotion_Aware_Facial_Animation
from jiwer import wer, mer, wil
from utils.lipnet import LipNet


RAIDROOT = os.environ['RAIDROOT']


def get_position(size, padding=0.25):

    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
         0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
         0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
         0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
         0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
         0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
         0.553364, 0.490127, 0.42689]

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
         0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
         0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
         0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
         0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
         0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
         0.784792, 0.824182, 0.831803, 0.824182]

    x, y = np.array(x), np.array(y)

    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def prepare_video(array, device, verbose=False):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device=device)
    points = [fa.get_landmarks(I) for I in array]

    front256 = get_position(256)
    video = []
    for point, scene in zip(points, array):
        if point is not None:
            shape = np.array(point[0])
            shape = shape[17:]
            M = transformation_from_points(
                np.matrix(shape), np.matrix(front256))

            img = cv2.warpAffine(scene, M[:2], (256, 256))
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160 // 2
            img = img[y - w // 2:y + w // 2, x - w:x + w, ...]
            img = cv2.resize(img, (128, 64))

            # Visualize
            if verbose:
                cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                1 / 0

            video.append(img)

    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0

    return video


def ctc_arr2txt(arr, start):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    pre = -1
    txt = []
    for n in arr:
        if(pre != n and n >= start):
            if(len(txt) > 0 and txt[-1] == ' ' and letters[n - start] == ' '):
                pass
            else:
                txt.append(letters[n - start])
        pre = n
    return ''.join(txt).strip()


def decode_sentence(y):
    y = y.argmax(-1)
    return ctc_arr2txt(y, start=1).lower()


def lipnet_predict(video, model):
    if video is None:
        return None
    y = model(video[None, ...].cuda())
    txt = decode_sentence(y[0])

    return txt


def read_transcript(file):
    with open(file, 'r') as f:
        text = f.readline()
    return text.lower()


def get_model(device):
    model = LipNet()
    model = model.to(device)

    pretrained_dict = torch.load(RAIDROOT + 'Networks/lipnet.pt')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items(
    ) if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items(
    ) if k not in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--model_type', type=str, default='net3')
    parser.add_argument('--audio_type', type=str, default='deepspeech')
    parser.add_argument('--audio_multiplier', type=float, default=2.0)
    parser.add_argument('--audio_truncation', type=float, default=0.8)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Init model
    model = Emotion_Aware_Facial_Animation(
        model_path=args.model_path,
        device=device,
        model_type=args.model_type,
        audio_type=args.audio_type,
        T=8
    )

    lipnet_model = get_model(device)

    dataset = args.dataset

    if os.path.exists(f'/home/meissen/Datasets/{dataset}/'):
        root_path = f'/home/meissen/Datasets/{dataset}/'
    else:
        root_path = RAIDROOT + f'Datasets/{dataset}/'
    latent_root = root_path + 'Aligned256/'
    transcript_root = root_path + 'Video/'

    videos = []
    with open(root_path + f'{dataset.lower()}_videos.txt', 'r') as f:
        line = f.readline()
        while line:
            videos.append(line.replace('\n', ''))
            line = f.readline()

    wer_sum = 0.
    mer_sum = 0.
    wil_sum = 0.
    for video in videos:
        latentfile = f"{latent_root}{video}/mean.latent.pt"
        sentence = f"{latent_root}{video}/"
        transcriptfile = f"{transcript_root}{video}.transcript.txt"
        # print(f"Image {imagefile} - Audio {audiofile} - Target {transcriptfile}")

        # Load transcript
        transcript = read_transcript(transcriptfile)

        # Create video
        max_sec = 30 if dataset == 'AudioDataset' else None
        max_sec = 1 if args.verbose else max_sec
        vid = model(test_latent=latentfile, test_sentence_path=sentence,
                    audio_multiplier=args.audio_multiplier,
                    audio_truncation=args.audio_truncation,
                    max_sec=max_sec)
        vid = (np.rollaxis(vid.numpy(), 1, 4) * 255.).astype(np.uint8)

        vid = prepare_video(vid, device, verbose=args.verbose)

        prediction = lipnet_predict(vid, lipnet_model)
        if prediction is None:
            continue
        transcript = read_transcript(transcriptfile)
        wer_error = wer(transcript, prediction)
        mer_error = mer(transcript, prediction)
        wil_error = wil(transcript, prediction)
        wer_sum += wer_error
        mer_sum += mer_error
        wil_sum += wil_error
        print(f"WER {wer_error:.4f} - MER {mer_error:.4f} - WIL {wil_error:.4f} prediction: {prediction} | transcript: {transcript}")

    print(f"Mean WER {wer_sum / len(videos):.4f}")
    print(f"Mean MER {mer_sum / len(videos):.4f}")
    print(f"Mean WIL {wil_sum / len(videos):.4f}")
