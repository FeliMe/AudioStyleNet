import argparse
import os
import pathlib
import torch


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--actor', type=str, required=True)
    parser.add_argument('--emotions',
                        help='two emotions to calculate the offset vector between',
                        nargs='+', required=True)
    args = parser.parse_args()

    # Read arguments
    path_to_actor = args.actor
    if path_to_actor[-1] == '/':
        path_to_actor = path_to_actor[:-1]

    assert len(args.emotions) == 2
    emotion1 = args.emotions[0]
    emotion2 = args.emotions[1]
    mapping = {
        'neutral': '01',
        'calm': '02',
        'happy': '03',
        'sad': '04',
        'angry': '05',
        'fearful': '06',
        'disgust': '07',
        'surprised': '08'
    }

    actor = path_to_actor.split('/')[-1]
    print("Computing offset between {} and {} of {}".format(
        emotion1, emotion2, actor))

    # Find sentences
    sentences = [str(f) for f in list(pathlib.Path(path_to_actor).glob('*'))]
    sentences = sorted(sentences)

    sentences1 = list(filter(lambda s: s.split(
        '/')[-1].split('-')[2] == mapping[emotion1], sentences))
    sentences2 = list(filter(lambda s: s.split(
        '/')[-1].split('-')[2] == mapping[emotion2], sentences))

    # Get frames
    frames1 = []
    for sentence in sentences1:
        for f in list(pathlib.Path(sentence).glob('*')):
            frames1.append(str(f))

    frames2 = []
    for sentence in sentences2:
        for f in list(pathlib.Path(sentence).glob('*')):
            frames2.append(str(f))

    frames1 = sorted(frames1)
    frames2 = sorted(frames2)

    # Load latent vectors
    vectors1 = torch.stack([torch.load(frame) for frame in frames1])
    vectors2 = torch.stack([torch.load(frame) for frame in frames2])

    # Compute mean offset
    offset_means = []
    for vector in vectors1:
        offset_means.append((vectors2 - vector).mean(dim=0))
    offset_means = torch.stack(offset_means)
    print(offset_means.shape)
    offset = offset_means.mean(dim=0)

    # Save
    os.makedirs('../saves/offsets/', exist_ok=True)
    torch.save(offset, '../saves/offsets/{}_offset_{}-{}.pt'.format(
        actor, emotion1, emotion2
    ))
