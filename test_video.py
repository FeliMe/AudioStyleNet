import argparse

from eafa import Emotion_Aware_Facial_Animation


parser = argparse.ArgumentParser()
parser.add_argument('--latentfile', type=str)
parser.add_argument('--sentence_path', type=str)
parser.add_argument('--audiofile', type=str)
parser.add_argument('--target_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--direction', default=None)
parser.add_argument('--max_sec', type=int, default=None)
parser.add_argument('--model_type', type=str, default='net3')
parser.add_argument('--audio_type', type=str, default='deepspeech')
parser.add_argument('--audio_multiplier', type=float, default=2.0)
parser.add_argument('--audio_truncation', type=float, default=0.8)
parser.add_argument('--direction_multiplier', type=float, default=1.0)
args = parser.parse_args()

device = f"cuda:{args.gpu}"

# Init model
model = Emotion_Aware_Facial_Animation(
    model_path=args.model_path,
    device=device,
    model_type=args.model_type,
    audio_type=args.audio_type,
    T=8,
    n_latent_vec=4,
    normalize_audio=False
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
