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


import os
import argparse

from glob import glob
from utils.audio_feature_extractor import audio_feature_extractor
from utils.audio_handler import AudioHandler

filename = './audio/merkel_2018_cut.wav'


parser = argparse.ArgumentParser(description='Voice operated character animation')
parser.add_argument('--ds_fname', default='./ds_graph/output_graph.pb', help='Path to trained DeepSpeech model')
# parser.add_argument('--audio_fname', default='./audio/test_sentence.wav', help='Path of input speech sequence')
parser.add_argument('--audiofiles', default=filename, help='Path of input speech sequence')
parser.add_argument('--out_path', default='./output', help='Output path')
parser.add_argument('--target_fps', default=25, help='Target frame rate')

args = parser.parse_args()
target_fps = float(args.target_fps)

print('OUT:', args.out_path)
print('FPS:', target_fps)

# Get AudioPaths
if os.path.isdir(args.audiofiles):
    audiofiles = glob(args.audiofiles + '*.wav')
else:
    audiofiles = [args.audiofiles]
assert len(audiofiles) > 0, f"Found no audiofiles in {args.audiofiles}"

# Init AudioHandler
config = {}
config['deepspeech_graph_fname'] = args.ds_fname
config['audio_feature_type'] = 'deepspeech'
config['num_audio_features'] = 29

config['audio_window_size'] = 16
config['audio_window_stride'] = 1

audio_handler = AudioHandler(config)

# Loop over found audiofiles
for audiofile in audiofiles:
    if args.out_path[-1] != '/':
        args.out_path += '/'
    out_path = args.out_path + audiofile.split('/')[-1].split('.')[0] + '/'
    os.makedirs(out_path, exist_ok=True)
    audio_feature_extractor(audio_handler, audiofile, target_fps, out_path)
