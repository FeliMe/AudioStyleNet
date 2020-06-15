'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''


import numpy as np
import time

from scipy.io import wavfile


def process_audio(audio_handler, audio, sample_rate, target_fps):
    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    return audio_handler.process(tmp_audio, target_fps)['subj']['seq']['audio']


def audio_feature_extractor(audio_handler, audio_fname, target_fps, out_path):
    print('Load audio file')
    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    print('Process audio')

    # torch.cuda.synchronize()
    a = time.perf_counter()
    processed_audio = process_audio(audio_handler, audio, sample_rate, target_fps)
    b = time.perf_counter()
    print('processing time:', 1000.0 * (b - a) / processed_audio.shape[0])

    print('audio processed type:', type(processed_audio))
    print('audio processed shape:', processed_audio.shape)

    num_frames = processed_audio.shape[0]
    print('num_frames:', num_frames)

    # Visualize
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # fig, ax = plt.subplots()
    # mfcc_data = np.swapaxes(processed_audio[0], 0, 1)
    # cax = ax.imshow(processed_audio[0], interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    # plt.show()

    for i in range(1, num_frames + 1):
        fname = out_path + '/' + str(i).zfill(5) + '.deepspeech'
        np.save(fname, processed_audio[i - 1])
        # np.savetxt(fname, processed_audio[i], delimiter=',')
