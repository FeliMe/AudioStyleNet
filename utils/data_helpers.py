import cv2
import dlib
import json
import numpy as np
import os
import pathlib
import pandas as pd
import subprocess
import sys
import torch

# from dreiDDFA.model import dreiDDFA
from glob import glob
from PIL import Image
from skimage import io
from torchvision import transforms
from tqdm import tqdm
from utils import get_mouth_params
from alignment_handler import AlignmentHandler


def align_videos(root_path, group):
    if root_path[-1] != '/':
        root_path += '/'

    aligner = AlignmentHandler()

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')
    videos = glob(root_path + '*.mp4')
    assert len(videos) > 0

    groups = []
    n = len(videos) // 7
    for i in range(0, len(videos), n):
        groups.append(videos[i:i + n])

    videos = groups[group]
    print(
        f"Group {group}, num_videos {len(videos)}, {len(groups)} groups in total")

    for i_video, video in enumerate(tqdm(videos)):
        vid_name = video.split('/')[-1][:-4]
        save_dir = os.path.join(target_path, vid_name)
        print("Video [{}/{}], {}".format(
            i_video + 1, len(videos),
            save_dir))

        aligner.align_video(video, save_dir)


def encode_frames(root_path):
    if root_path[-1] != '/':
        root_path += '/'

    videos = sorted(glob(root_path + '*/'))
    videos = [sorted(glob(v + '*.png')) for v in videos]
    all_frames = [item for sublist in videos for item in sublist]
    assert len(all_frames) > 0
    print(len(all_frames))

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoder
    from my_models.models import resnetEncoder
    e = resnetEncoder(net=18).eval().to(device)
    e.load_state_dict(torch.load("PATH_HERE"))

    # Get latent avg
    from my_models.style_gan_2 import PretrainedGenerator1024
    g = PretrainedGenerator1024().eval()
    latent_avg = g.latent_avg.view(1, -1).repeat(18, 1)

    # transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for frame in tqdm(all_frames):
        save_path = frame.split('.')[0] + '.latent.pt'
        # print(save_path)
        if os.path.exists(save_path):
            continue

        # Load image
        img = t(Image.open(frame)).unsqueeze(0).to(device)

        # Encoder image
        with torch.no_grad():
            latent_offset = e(img)[0].cpu()
            latent = latent_offset + latent_avg

        # Visualize
        from torchvision.utils import make_grid
        from utils.utils import downsample_256
        print(save_path, latent.shape)
        img_gen = g.to(device)([latent.to(device)],
                               input_is_latent=True, noise=g.noises)[0].cpu()
        img_gen = downsample_256(img_gen)
        img_gen = make_grid(torch.cat((img_gen, img.cpu()),
                                      dim=0), normalize=True, range=(-1, 1))
        img_gen = transforms.ToPILImage('RGB')(img_gen)
        img_gen.show()
        1 / 0

        # Save
        # torch.save(latent, save_path)


def get_mean_latents(root):
    # Load paths
    videos = sorted(glob(root + '*/'))

    for video in tqdm(videos):
        latent_paths = sorted(glob(video + '*.latent.pt'))

        mean_latent = []
        for latent_path in latent_paths:
            latent = torch.load(latent_path).unsqueeze(0)
            mean_latent.append(latent)
        mean_latent = torch.cat(mean_latent, dim=0).mean(dim=0)

        # Save
        torch.save(mean_latent, video + 'mean.latent.pt')


def get_landmarks(root_path, group):
    detector_path = '/home/meissen/Datasets/mmod_human_face_detector.dat'
    detector = dlib.cnn_face_detection_model_v1(detector_path)
    predictor_path = '/home/meissen/Datasets/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    if root_path[-1] != '/':
        root_path += '/'

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')
    videos = sorted(glob(root_path + '*/'))
    assert len(videos) > 0

    groups = []
    n = len(videos) // 6
    for i in range(0, len(videos), n):
        groups.append(videos[i:i + n])

    videos = groups[group]
    print(
        f"Group {group}, num_videos {len(videos)}, {len(groups)} groups in total")

    for video in tqdm(videos):
        print(video)
        frames = sorted(glob(video + '*.png'))

        for frame in frames:
            save_path = frame.split('.')[0] + '.landmarks.pt'
            img = io.imread(frame)

            # Grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Detect faces
            rects = detector(img, 1)
            if len(rects) == 0:
                print(f"Did not detect a face in {frame}")
            rect = rects[0]
            landmarks = torch.tensor([(item.x, item.y) for item in predictor(
                gray, rect).parts()], dtype=torch.float32)

            # Visualize
            # print(save_path)
            # print(landmarks.shape)
            # for (x, y) in landmarks:
            #     cv2.circle(img, (x, y), 1, (255, 255, 255), 1)
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            # 1 / 0

            # Save
            torch.save(landmarks, save_path)


def get_scores(root_path, model='fer'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model == 'fer':
        from my_models.models import FERClassifier
        model = FERClassifier(softmaxed=False).eval().to(device)
        appendix = '-logit_fer.pt'
    elif model == 'ravdess':
        from my_models.models import EmotionClassifier
        model = EmotionClassifier(softmaxed=False).eval().to(device)
        appendix = '-logit_ravdess.pt'
    else:
        raise NotImplementedError

    if root_path[-1] == '/':
        root_path = root_path[:-1]

    t = transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    videos = sorted(glob(root_path + '*/'))
    for video in videos:
        frames = sorted(glob(video + '*.png'))
        for frame in frames:
            save_path = frame.split('.')[0] + appendix

            img = t(Image.open(frame)).unsqueeze(0).to(device)
            logits = model(img)[0].cpu()
            print(save_path)
            print(logits)
            1 / 0

            torch.save(logits, save_path)


def omg_get_forced_alignment(root_path):
    all_audios = [str(a) for a in list(pathlib.Path(root_path).glob('*/*.wav'))]

    transcript_path = [str(a) for a in list(
        pathlib.Path(root_path).glob('*_transcripts.csv'))][0]
    transcripts = pd.read_csv(transcript_path)

    print(transcripts)
    print(len(all_audios))

    for audio in tqdm(all_audios):
        video, utterance = audio.split('/')[-2:]
        forced_alignment_path = audio[:-4] + '.json'
        if os.path.exists(forced_alignment_path):
            continue
        utterance = utterance[:-4] + '.mp4'
        transcript = transcripts.loc[(
            transcripts.video == video) & (transcripts.utterance == utterance)].transcript.to_list()[0]
        transcript = str(transcript)
        # print(transcript)
        # print(audio)
        # print(video, utterance)
        # print(forced_alignment_path)
        # 1 / 0
        print(audio, transcript)
        t_path = '../saves/tmp.txt'
        with open(t_path, 'w') as f:
            f.write(transcript)

        command = f'curl -F "audio=@{audio}" -F "transcript=@{t_path}" "http://localhost:8765/transcriptions?async=false"'

        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        # out = proc.stdout.read()
        out, err = proc.communicate()
        aligmnent = json.loads(out)

        with open(forced_alignment_path, 'w') as f:
            json.dump(aligmnent, f)


def omg_get_phoneme_timing(root_path):
    all_aligned = [str(a)
                   for a in list(pathlib.Path(root_path).glob('*/*.json'))]

    # aligned256_root = '/'.join(root_path.split('/')[:-3] + ['Aligned256'] + root_path.split('/')[-2:])

    info_path = [str(a) for a in list(
        pathlib.Path(root_path).glob('*_info.csv'))][0]
    info = pd.read_csv(info_path)

    print(len(all_aligned))

    results = {}
    for aligned in tqdm(all_aligned):

        # Get corresponding frame sequence
        target_path = '/'.join(aligned.split('/')
                               [:-4] + ['Aligned256'] + aligned.split('/')[-3:])[:-5] + '/'
        target_frames = sorted([str(p) for p in list(
            pathlib.Path(target_path).glob('*.png'))])
        print(len(target_frames), target_path)
        maxi = int(target_frames[-1].split('/')[-1].split('.')[0])

        # Get relevant paths
        video, utterance = aligned.split('/')[-2:]
        utterance = utterance[:-5] + '.mp4'

        # Get FPS
        fps = float(info.loc[(info['video'] == video) & (
            info['utterance'] == utterance)].fps.to_list()[0])

        with open(aligned) as json_file:
            data = json.load(json_file)
        print(aligned)
        print(data['transcript'])
        # print("")
        words = data['words']

        # Remove failed words
        new_words = []
        for i_word in range(len(words)):
            if 'start' in words[i_word].keys():
                new_words.append(words[i_word])
        words = new_words

        # Fill missing silence words
        words_new = []
        for i_word in range(len(words)):
            if i_word != 0 and words[i_word - 1]['end'] < words[i_word]['start']:
                word = {
                    'alignedWord': 'sil',
                    'end': words[i_word]['start'],
                    'phones': [
                        {
                            'duration': round(words[i_word]['start'] - words[i_word - 1]['end'], 2),
                            'phone': 'sil'
                        }],
                    'start': words[i_word - 1]['end'],
                    'word': 'sil'
                }
                words_new.append(word)
            words_new.append(words[i_word])
        words = words_new

        # Get absolute timings for each phoneme
        for word in words:
            phones = word['phones']
            for i in range(len(phones)):
                if i == 0:
                    phones[i]['abs_start'] = word['start']
                    phones[i]['abs_end'] = round(
                        phones[i]['abs_start'] + phones[i]['duration'], 2)
                else:
                    phones[i]['abs_start'] = phones[i - 1]['abs_end']
                    phones[i]['abs_end'] = round(
                        phones[i]['abs_start'] + phones[i]['duration'], 2)

        # Get timings in a dict
        timings = {}
        timings_f = {}
        for i_word in range(len(words)):
            # print(words[i_word]['word'])
            for phone in words[i_word]['phones']:
                timings[phone['abs_start']] = phone['phone']
                timings_f[phone['abs_start'] // (1. / fps)] = phone['phone']
                # print(phone['abs_start'], phone['abs_start'] // (1. / fps), timings[phone['abs_start']])
            if i_word == len(words) - 1:
                timings[phone['abs_end']] = 'sil'
                timings_f[phone['abs_end'] // (1. / fps)] = 'sil'
                # print(phone['abs_end'], phone['abs_end'] // (1. / fps), timings[phone['abs_end']])
            # print("")

        # Assign Phonemes to frames
        phone = 'sil'
        phones = []
        for i in range(maxi):
            target_frame = target_path + str(i + 1).zfill(3) + '.png'
            if i in timings_f.keys():
                phone = timings_f[i]
            phones.append(phone)
            # print(target_frame, i, phone)
            results[target_frame] = phone

        print("")

    with open(root_path + 'omg_mapping_phoneme.json', 'w') as f:
        json.dump(results, f)


def omg_extract_face_feats(root_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        '/home/meissen/Datasets/RAVDESS/shape_predictor_68_face_landmarks.dat')

    if root_path[-1] != '/':
        root_path += '/'
    save_path = root_path + 'omg_mouth_features.npy'

    videos = [str(v) for v in list(pathlib.Path(root_path).glob('*/*/'))]

    all_paths = [sorted([str(p) for p in list(pathlib.Path(v).glob('*.png'))])
                 for v in videos]

    result = {}
    result = np.load(save_path, allow_pickle=True).item()

    counter = 0
    for v in tqdm(all_paths):
        for f in v:
            video, utterance, n_frame = f.split('/')[-3:]
            key = f"{video}/{utterance}/{n_frame}"
            if key in result.keys():
                continue
            # Load image
            frame = cv2.imread(f)

            # Grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect faces
            rects = detector(frame, 1)
            if len(list(rects)) == 0:
                print(f"WARNING. NO FACE DETECTED IN {f}")
                counter += 1
            for rect in rects:
                landmarks = [(int(item.x), int(item.y))
                             for item in predictor(gray, rect).parts()]
                landmarks = np.array(landmarks)

                params = get_mouth_params(landmarks, frame)

                # Visualize
                # for (x, y) in landmarks:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.imshow("Output", frame)
                # cv2.waitKey(0)
                # print(key)
                # 1 / 0

                result[key] = params
                break

    # Save params
    print(f"NO FACES DETECTED IN {counter} FRAMES")
    print("Done. Saving...")
    np.save(save_path, result)


def tagesschau_gather_info(root):
    save_path = root + 'latent_data.pt'
    frames = sorted(glob(root + '*/*.png'))

    names = []
    latents = []
    landmarks = []
    deepspeechs = []
    for frame in tqdm(frames):
        base = frame.split('.')[0]
        name = '/'.join(base.split('/')[-2:])
        latent = torch.load(base + '.latent.pt')
        landmark = torch.load(base + '.landmarks.pt')
        deepspeech = torch.tensor(
            np.load(base + '.deepspeech.npy'), dtype=torch.float32)

        names.append(name)
        latents.append(latent)
        landmarks.append(landmark)
        deepspeechs.append(deepspeech)

    names = np.array(names)
    latents = torch.stack(latents)
    landmarks = torch.stack(landmarks)
    deepspeechs = torch.stack(deepspeechs)

    data = {
        'names': names,
        'latents': latents,
        'landmarks': landmarks,
        'deepspeech': deepspeechs
    }

    torch.save(data, save_path)


def get_3ddfa_params(root_path):
    device = 'cuda'

    model = dreiDDFA(True).to(device)

    videos = sorted(glob(root_path + '*/'))
    frames = [sorted(glob(v + '*.png')) for v in videos]
    frames = [item for sublist in frames for item in sublist]

    t = transforms.ToTensor()

    for frame in tqdm(frames):
        save_path = frame.split('.')[0] + '.3ddfa.pt'
        if os.path.exists(save_path):
            continue

        img = t(Image.open(frame)).unsqueeze(0).to(device)
        res = model.predict_param(img)
        if res is None:
            print(f"No face detected in {frame}")
            continue

        params = {
            'param': res['param'][0].cpu(),
            'roi_box': res['roi_box']
        }

        # Visualize
        # print(params['param'].shape)
        # print(params['roi_box'])
        # print(save_path)
        # from dreiDDFA.ddfa_utils import draw_landmarks
        # landmarks = model.reconstruct_vertices(res['param'], res['roi_box'], dense=False)
        # draw_landmarks(img[0].cpu(), landmarks[0].cpu(), show_flg=True)
        # 1 / 0

        # Save
        torch.save(params, save_path)


if __name__ == "__main__":

    path = sys.argv[1]
    align_videos(path, int(sys.argv[2]))


"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

"""
Download files from google drive

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=CODE_FROM_ABOVE&id=FILEID'

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pQy1YUGtHeUM2dUE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=&id=0B7EVK8r0v71pQy1YUGtHeUM2dUE'
0B7EVK8r0v71pQy1YUGtHeUM2dUE
"""
