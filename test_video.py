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
    T=8
)

# Create video
vid = model(test_latent=args.latentfile, test_sentence_path=args.sentence_path,
            direction=args.direction,
            audio_multiplier=args.audio_multiplier,
            audio_truncation=args.audio_truncation,
            direction_multiplier=args.direction_multiplier,
            max_sec=args.max_sec)

# from torchvision import transforms
# import os
# save_dir = f'/home/meissen/{args.latentfile.split("/")[-2]}_trunc_{args.audio_truncation}_mult_{args.audio_multiplier}/'
# os.makedirs(save_dir, exist_ok=True)
# for i, frame in enumerate(vid[:-1]):
#     frame = transforms.ToPILImage('RGB')(frame)
#     frame.save(f'{save_dir}{str(i + 1).zfill(5)}.png')

# Save video
# import os
# args.target_path = os.path.join("/home/meissen/", f"{args.latentfile.split('/')[-2]}_{args.direction.split('/')[-1].split('.')[0]}_{args.direction_multiplier}.avi")
model.save_video(vid, args.audiofile, args.target_path)
