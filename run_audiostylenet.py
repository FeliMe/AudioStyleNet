"""
Test model by generating a video from an encoded still image in 'latentfile'
and the extracted deepspeech features in 'sentence_path'. Overlay the generated
video with the original audio from 'audiofile'. Optionally: add a further
manipulation from 'direction'.
"""

import argparse
import os

from audiostylenet import AudioStyleNet


parser = argparse.ArgumentParser()
parser.add_argument('--latentfile', type=str, default='data/images/yt_xOpJdHiIwhQ_2.latent.pt')
parser.add_argument('--sentence_path', type=str, default='data/audio/camila/')
parser.add_argument('--audiofile', type=str, default='data/audio/camila/camila.mp3')
parser.add_argument('--target_path', type=str, default='./output/demo01.avi')
parser.add_argument('--model_path', type=str, default='model/audiostylenet.pt')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--direction', default=None)
parser.add_argument('--max_sec', type=int, default=None)
parser.add_argument('--audio_type', type=str, default='deepspeech')
parser.add_argument('--audio_multiplier', type=float, default=2.0)
parser.add_argument('--audio_truncation', type=float, default=0.8)
parser.add_argument('--direction_multiplier', type=float, default=1.0)
args = parser.parse_args()

# Check if target directory exists
if not os.path.exists(os.path.dirname(args.target_path)):
    os.makedirs(os.path.dirname(args.target_path), exist_ok=True)

device = f"cuda:{args.gpu}"

# Init model
model = AudioStyleNet(
    model_path=args.model_path,
    device=device,
    audio_type=args.audio_type,
    T=8
)

# Create video
vid = model(test_latent=args.latentfile, test_sentence_path=args.sentence_path,
            direction=args.direction,
            audio_multiplier=args.audio_multiplier,
            audio_truncation=args.audio_truncation,
            direction_multiplier=args.direction_multiplier,
            max_sec=args.max_sec)

# Save video
model.save_video(vid, args.audiofile, args.target_path)
