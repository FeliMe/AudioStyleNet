import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from matplotlib.widgets import Slider, Button
from my_models import models
from my_models.style_gan_2 import Generator
from pathlib import Path
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from tqdm import tqdm

EMOTIONS = ['neutral', 'calm', 'happy', 'sad',
            'angry', 'fearful', 'disgusted', 'surprised']


def int_to_one_hot(labels):
    one_hots = []
    for label in labels:
        one_hot = torch.zeros(8)
        one_hot[label] = 1
        one_hots.append(one_hot)
    return torch.stack(one_hots, dim=0)


def downsample_256(img):
    b, c, h, w = img.shape
    factor = h // 256
    img = img.reshape(b, c, h // factor, factor, w // factor, factor)
    img = img.mean([3, 5])
    return img


def genereate_training_data(num_samples):

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)

    # Init Classifier
    rav = models.EmotionClassifier().eval().to(device)
    fer = models.FERClassifier().to(device)

    # Generate images
    zs = []
    latents = []
    scores_fer = []
    scores_rav = []
    for i in tqdm(range(num_samples // 10)):
        with torch.no_grad():
            z = torch.randn((10, 512), device=device)
            latent = g.style(z).view(-1, 1, g.style_dim)
            img, _ = g([latent], input_is_latent=True, truncation=0.85,
                       truncation_latent=g.latent_avg.to(device))
            img = downsample_256(img)
            score_fer = fer(img).cpu()
            score_rav = rav(img).cpu()
            z = z.cpu()
            latent = latent.cpu()
            img = img.cpu()
        zs.append(z)
        latents.append(latent)
        scores_fer.append(score_fer)
        scores_rav.append(score_rav)

    zs = torch.cat(zs, dim=0)
    latents = torch.cat(latents, dim=0)
    scores_fer = torch.cat(scores_fer, dim=0)
    scores_rav = torch.cat(scores_rav, dim=0)

    # Some info
    import matplotlib.pyplot as plt
    emotions = ['neutral', 'calm', 'happy', 'sad',
                'angry', 'fearful', 'disgusted', 'surprised']
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for i, e in enumerate(emotions):
        emo_fer_sorted, _ = torch.sort(scores_fer[:, i])
        emo_rav_sorted, _ = torch.sort(scores_rav[:, i])

        ax_x = i // 4
        ax_y = i % 4

        axs[ax_x, ax_y].plot(emo_fer_sorted, label='fer')
        axs[ax_x, ax_y].plot(emo_rav_sorted, label='rav')
        axs[ax_x, ax_y].set_title(e)
        axs[ax_x, ax_y].legend()
    # plt.show()
    plt.savefig(f'saves/control_latent/latent_training_data_distribution_{num_samples}.png')

    data = {
        'zs': zs,
        'latents': latents,
        'scores_fer': scores_fer,
        'scores_rav': scores_rav
    }

    torch.save(data, f'saves/control_latent/latent_training_data_{num_samples}.pt')


def find_direction(args):

    # Load training data
    data = torch.load(args.training_data)
    latents = data['latents']
    scores_rav = data['scores_rav']
    scores_fer = data['scores_fer']

    MAPPING = {
        'neutral': 0,
        'calm': 1,
        'happy': 2,
        'sad': 3,
        'angry': 4,
        'fearful': 5,
        'disgusted': 6,
        'surprised': 7
    }

    emotion = 'happy'

    X = latents.numpy().reshape((-1, 512))
    y = scores_rav[:, MAPPING[emotion]].numpy()
    # y = scores_fer[:, MAPPING[emotion]].numpy()

    # Make binary labels
    y = np.round(y)

    # Balance classes
    non_zeros = np.count_nonzero(y)
    sort_inds = y.argsort()
    y = y[sort_inds][max(0, len(y) - 2 * non_zeros):]
    X = X[sort_inds][max(0, len(X) - 2 * non_zeros):]
    print(f"Samples left: {len(X)}")

    # Shuffle data
    shuffle_inds = np.arange(y.shape[0])
    np.random.shuffle(shuffle_inds)
    y = y[shuffle_inds]
    X = X[shuffle_inds]

    # Train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y)
    print(clf.coef_.shape)
    direction = clf.coef_.reshape((1, 512))

    # Save direction
    np.save(
        f'saves/control_latent/directions/{emotion}_rav_lin.npy', direction * 2.0)


def control_latent_video(args):

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    latent_name = args.input_latent.split('/')[-2]
    vec_type = args.vec.split('/')[-1].split('.')[0]

    # Load direction
    direction = torch.tensor(np.load(args.vec), dtype=torch.float32).to(device)

    # Get original latents
    paths = sorted([str(p) for p in list(Path(args.input_latent).glob('*.pt'))])

    # Init generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)
    g.noises = [n.to(device) for n in g.noises]

    save_dir = 'saves/control_latent/videos/'
    tmp_dir = save_dir + '.temp/'
    os.makedirs(tmp_dir, exist_ok=True)
    for i, p in tqdm(enumerate(paths)):
        latent = torch.load(p).to(device)
        latent[:8] = (latent + 2 * direction)[:8]

        img, _ = g([latent], input_is_latent=True, noise=g.noises)

        save_image(img, tmp_dir + str(i + 1).zfill(3) + '.png',
                   normalize=True, range=(-1, 1))
        # img = make_grid(img.cpu(), normalize=True, range=(-1, 1))
        # img = transforms.ToPILImage('RGB')(img)
        # img.show()
        # break

    # Convert output frames to video
    original_dir = os.getcwd()
    os.chdir(tmp_dir)
    name = latent_name + '_' + vec_type
    os.system(f'ffmpeg -framerate 30 -i %03d.png -c:v libx264 -r 30 -pix_fmt yuv420p ../{name}.mp4')

    # Remove generated frames and keep only video
    os.chdir(original_dir)
    os.system(f'rm -r {tmp_dir}')


def control_latent(args):
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    latent_name = args.input_latent.split('/')[-1].split('.')[0]
    vec_type = args.vec.split('/')[-1].split('.')[0]

    # Load vector
    vec = torch.tensor(np.load(args.vec), dtype=torch.float32).to(device)

    # Init Generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)
    g.noises = [n.to(device) for n in g.noises]

    if args.input_latent == 'random':
        input_latent = g.get_latent(
            inp=[torch.randn(1, 512, device=device)],
            truncation=0.5,
            truncation_latent=g.latent_avg.to(device)
        )[0].unsqueeze(0)
        input_latent = input_latent.repeat(1, g.n_latent, 1)
    else:
        input_latent = torch.load(args.input_latent).to(device)

    n_steps = 120
    coeffs = np.linspace(-1, 1, num=n_steps)
    # coeffs = np.linspace(0, 1, num=n_steps)

    tmp_dir = args.save_dir + '.temp/'
    os.makedirs(tmp_dir, exist_ok=True)
    for i, coeff in tqdm(enumerate(coeffs)):
        with torch.no_grad():
            latent = input_latent.clone()
            latent[:8] = (latent + coeff * vec)[:8]
            img, _ = g([latent], input_is_latent=True, noise=g.noises)

            save_image(img, tmp_dir + str(i + 1).zfill(3) +
                       '.png', normalize=True, range=(-1, 1))

    # Convert output frames to video
    original_dir = os.getcwd()
    os.chdir(tmp_dir)
    name = latent_name + '_' + vec_type
    os.system(
        f'ffmpeg -framerate 30 -i %03d.png -c:v libx264 -r 30 -pix_fmt yuv420p ../{name}.mp4')

    # Remove generated frames and keep only video
    os.chdir(original_dir)
    os.system(f'rm -r {tmp_dir}')


def demo():
    """
    Matplotlib slideshow
    """

    # Load directions
    directions_path = 'saves/control_latent/directions/'
    directions = np.array([
        np.load(directions_path + 'neutral_rav_lin.npy'),
        np.load(directions_path + 'calm_rav_lin.npy'),
        np.load(directions_path + 'happy_rav_lin.npy'),
        np.load(directions_path + 'sad_rav_lin.npy'),
        np.load(directions_path + 'angry_rav_lin.npy'),
        np.load(directions_path + 'fearful_rav_lin.npy'),
        np.load(directions_path + 'disgusted_rav_lin.npy'),
        np.load(directions_path + 'surprised_rav_lin.npy'),
    ]).reshape(8, -1)

    # Load input images
    input_latents = torch.stack([
        torch.load('saves/projected_images/obama.pt'),
        torch.load('saves/projected_images/generated.pt'),
        torch.load('saves/projected_images/pearl_earring.pt'),
        torch.load('saves/projected_images/felix.pt'),
        torch.load('saves/projected_images/neutral.pt'),
        torch.load('saves/projected_images/einstein.pt'),
    ])

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # To device
    input_latents = input_latents.to(device)
    directions = torch.tensor(directions, device=device)
    # nrow = input_latents.shape[0] // 2
    nrow = 3

    # Init generator
    g = Generator(1024, 512, 8, pretrained=True).eval().to(device)
    g.noises = [n.to(device) for n in g.noises]

    img, _ = g([input_latents], input_is_latent=True, noise=g.noises)
    img = downsample_256(img)
    img = torch.stack([make_grid(i, normalize=True, range=(-1, 1))
                       for i in img])
    img = make_grid(img, nrow=nrow)
    img = transforms.ToPILImage(mode='RGB')(img.cpu())

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.55)
    im = plt.imshow(img)
    plt.axis('off')
    ax_sliders = [plt.axes([0.2, 0.1 + 0.05 * i, 0.65, 0.03],
                           facecolor='lightgoldenrodyellow') for i in range(len(directions))]
    sliders = [Slider(ax_slider, emotion, 0.0, 1.0, valinit=0, valstep=0.01)
               for emotion, ax_slider in zip(EMOTIONS, ax_sliders)]

    def update(val):
        coeffs = torch.tensor(
            [slider.val for slider in sliders]).view(-1, 1).to(device)
        # Encode
        with torch.no_grad():
            latent = input_latents.clone()
            new_direction = (coeffs * directions).sum(dim=0, keepdim=True)
            # print(coeffs)
            latent[:, :8] = (latent + new_direction)[:, :8]

            # Decode
            img, _ = g([latent], input_is_latent=True, noise=g.noises)

            # Downsample to 256 x 256
            img = downsample_256(img)

        # Update plot
        img = torch.stack(
            [make_grid(i, normalize=True, range=(-1, 1)) for i in img])
        img = make_grid(img, nrow=nrow)
        img = transforms.ToPILImage(mode='RGB')(img.cpu())
        im.set_data(img)
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color='lightgoldenrodyellow',
                    hovercolor='0.975')

    def reset(event):
        for slider in sliders:
            slider.reset()

    button.on_clicked(reset)
    plt.show()


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--find_direction', action='store_true')
    parser.add_argument('--control_latent', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('-i', '--input_latent', type=str,
                        default='saves/projected_images/generated.pt')
    parser.add_argument('-v', '--vec', type=str,
                        default='saves/control_latent/directions/smile.npy')
    parser.add_argument('-d', '--save_dir', type=str,
                        default='saves/control_latent/videos/')
    parser.add_argument('--training_data', type=str,
                        default='saves/control_latent/latent_training_data_20000.pt')
    args = parser.parse_args()

    if args.generate_data:
        genereate_training_data(20000)
    elif args.find_direction:
        find_direction(args)
    elif args.control_latent:
        if os.path.isdir(args.input_latent):
            control_latent_video(args)
        else:
            control_latent(args)
    elif args.demo:
        demo()
    else:
        raise NotImplementedError
