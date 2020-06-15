import argparse
import os

from glob import glob
from utils.utils import VideoAligner2


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--filetype', type=str, choices=['image', 'video'], default='image')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


# Load target image
path = args.files
if os.path.isdir(path):
    if args.filetype == 'image':
        files = glob(path + '*.png')
        files += glob(path + '*.jpg')
    elif args.filetype == 'video':
        files = glob(path + '*.mp4')
        files += glob(path + '*.avi')
    else:
        raise NotImplementedError
else:
    files = [path]

# Select filetype
ext = files[0].split('.')[-1]
print(ext)
if ext.lower() in ['avi', 'mp4', 'flv', 'mpg']:
    filetype = 'video'
elif ext.lower() in ['jpg', 'jpeg', 'png']:
    filetype = 'image'
else:
    print("Unknown file type")
    raise NotImplementedError

aligner = VideoAligner2(device=f'cuda:{args.gpu}')
if not os.path.exists(args.out_dir):
    os.makedirs(args.outdir, exist_ok=True)

for file in files:
    if filetype == 'image':
        save_path = args.out_dir + file.split('/')[-1].split('.')[0] + '.png'
        print(f"Saving to {save_path}")
        aligner.align_single_image(file, save_path)
    else:
        save_dir = os.path.join(args.out_dir, file.split('/')[-1].split('.')[0])
        print(f"Saving to {save_dir}")
        aligner.align_video(file, save_dir)
