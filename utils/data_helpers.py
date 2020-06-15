import os
import sys
import torch

from glob import glob
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utils import VideoAligner


RAIDROOT = os.environ['RAIDROOT']


def align_videos(root_path, group):
    if root_path[-1] != '/':
        root_path += '/'

    aligner = VideoAligner()

    target_path = ('/').join(root_path.split('/')[:-2]) + '/Aligned256/'
    print(f'Saving to {target_path}')
    videos = glob(root_path + '*.mp4')
    assert len(videos) > 0

    groups = []
    n = len(videos) // 2
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
    # checkpoint = torch.load("PATH_HERE", map_location=device)
    checkpoint = torch.load(
        "/mnt/sdb1/meissen/Networks/GRID_new.pt", map_location=device)
    if type(checkpoint) == dict:
        e.load_state_dict(checkpoint['model'])
    else:
        e.load_state_dict(checkpoint)

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
        img_gen = g.to(device)([latent.unsqueeze(0).to(device)],
                               input_is_latent=True, noise=g.noises)[0].cpu()
        img_gen = downsample_256(img_gen)
        img_gen = make_grid(torch.cat((img_gen, img.cpu()),
                                      dim=0), normalize=True, range=(-1, 1))
        img_gen = transforms.ToPILImage('RGB')(img_gen)
        img_gen.show()
        1 / 0

        # Save
        torch.save(latent, save_path)


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


if __name__ == "__main__":

    path = sys.argv[1]
    # align_videos(path, int(sys.argv[2]))
    encode_frames(path)


"""
File with helper functions to modify datasets. Mostly those functions are
only used once.
"""

"""
Download files from google drive

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=CODE_FROM_ABOVE&id=FILEID'

wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p'
wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm=UzHO&id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc'
1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc
"""
