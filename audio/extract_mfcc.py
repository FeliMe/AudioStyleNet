import argparse
import numpy as np
import os
import python_speech_features

from glob import glob
from resampy import resample
from scipy.io import wavfile
from tqdm import tqdm


def read_file(file, target_sr):
    sr, audio = wavfile.read(file)
    if audio.ndim != 1:
        audio = audio[:, 0]

    return resample(audio.astype(float), sr, 16000).astype('int16')


def extract_mfcc(audio, sr, numcep=32, winstep=0.01, winlen=0.02, audio_window_size=0.64, fps=25):
    audio_window_size = int(audio_window_size / winstep)

    mfcc = python_speech_features.mfcc(
        audio, samplerate=sr, numcep=numcep, winlen=winlen, winstep=winstep, nfilt=max(26, numcep))

    # Make windows
    zero_pad = np.zeros((audio_window_size // 2, mfcc.shape[1]))
    mfcc = np.concatenate((zero_pad, mfcc, zero_pad), axis=0)
    windows = []
    for window_index in range(0, mfcc.shape[0] - audio_window_size, int((1 / winstep) / fps)):
        windows.append(mfcc[window_index:window_index + audio_window_size])

    return np.array(windows)


def plot_mfcc(window):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    ig, ax = plt.subplots()
    mfcc_data = np.swapaxes(window, 0, 1)
    ax.imshow(mfcc_data, interpolation='nearest',
              cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    # Showing mfcc_data
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str,
                        default='/home/meissen/Datasets/AudioDataset/Audio/')
    parser.add_argument('--output_dir', type=str,
                        default='/home/meissen/Datasets/AudioDataset/Aligned256/')
    parser.add_argument('--fps', type=int, default=25),
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()

    fps = args.fps
    sr = args.sample_rate

    files = sorted(glob(args.audio_dir + '*.wav'))

    for file in tqdm(files):
        save_dir = args.output_dir + file.split('/')[-1].split('.')[0] + '/'
        os.makedirs(save_dir, exist_ok=True)
        audio = read_file(file, sr)

        windows = extract_mfcc(audio, sr=sr, numcep=32,
                               winstep=0.01, winlen=0.02, audio_window_size=0.64, fps=fps)

        # Visualize
        print(f"{save_dir} Audio length {(len(audio) / sr) * fps:.2f} frames")
        print(windows.shape, windows.min(), windows.max(), windows.mean())
        plot_mfcc(windows[32])
        break

        for i in range(len(windows)):
            save_path = save_dir + f"{str(i + 1).zfill(5)}.mfcc.npy"
            np.save(save_path, windows[i])
