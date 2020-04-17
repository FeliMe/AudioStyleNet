import argparse
import librosa
import numpy as np
import os
import python_speech_features
import torch
import torchaudio

from glob import glob
from lpctorch import LPCCoefficients
from resampy import resample
from scipy.io import wavfile
from tqdm import tqdm


def read_file(file, target_sr):
    sr, audio = wavfile.read(file)
    # audio, sr = torchaudio.load(file, channels_first=False)
    # audio = audio.numpy()
    if audio.ndim != 1:
        audio = audio[:, 0]

    return resample(audio.astype(float), sr, 16000)


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


def plot_lpc(window):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    ig, ax = plt.subplots()
    lpc_data = np.swapaxes(window, 0, 1)
    ax.imshow(lpc_data, interpolation='nearest',
              cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('LPC')
    # Showing lpc_data
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str,
                        default='/home/meissen/Datasets/TestSentences/Obama/')
    parser.add_argument('--output_dir', type=str,
                        default='/home/meissen/Datasets/TestSentences/Obama/Aligned256/')
    parser.add_argument('--fps', type=int, default=25),
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()

    fps = args.fps
    sr = args.sample_rate

    n_features = 64
    n_coeffs = 32
    window_length = 0.02
    window_stride = 0.01

    lpc_prep = LPCCoefficients(
        sr,
        window_length,
        window_stride / window_length,
        order=(n_coeffs - 1)
    )

    files = sorted(glob(args.audio_dir + '*.wav'))
    print(f"Found {len(files)} wav-files")

    for file in tqdm(files):
        save_dir = args.output_dir + file.split('/')[-1].split('.')[0] + '/'
        audio = read_file(file, sr)

        # Compute lpcs
        lpcs = lpc_prep(torch.tensor(audio).unsqueeze(0))[0].numpy()

        # Pad zeros
        zero_pad = np.zeros((n_features // 2, lpcs.shape[1]))
        lpcs = np.concatenate((zero_pad, lpcs, zero_pad), axis=0)

        # Convert to windows with stride 40ms
        audio_windows = librosa.util.frame(
            lpcs, frame_length=64, hop_length=4, axis=0)

        # Visualize
        print(save_dir)
        print(audio_windows.shape, audio_windows.min(), audio_windows.max(), audio_windows.mean())
        plot_lpc(audio_windows[32])
        break

        # Save
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(audio_windows)):
            save_path = save_dir + f"{str(i + 1).zfill(5)}.lpc.npy"
            np.save(save_path, audio_windows[i])
