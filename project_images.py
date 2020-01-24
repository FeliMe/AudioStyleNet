import glob
import os
import sys
import torch

from tqdm import tqdm
from projector import Projector
from my_models.style_gan_2 import Generator
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


if __name__ == "__main__":

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    g = Generator(1024, 512, 8, pretrained=True).to(device).train()
    for param in g.parameters():
        param.requires_grad = False

    proj = Projector(g)

    # Load target image
    path = sys.argv[1]
    if os.path.isdir(path):
        image_files = glob.glob(path + '*.png')
        image_files += glob.glob(path + '*.jpg')
    else:
        image_files = [path]

    for i, file in tqdm(enumerate(sorted(image_files))):
        print('Projecting {}'.format(file))

        # Load image
        target_image = Image.open(file)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        target_image = transform(target_image).to(device)

        # Run projector
        proj.run(target_image, 1000 if i == 0 else 50)

        # Collect results
        generated = proj.get_images()
        latents = proj.get_latents()

        # Save results
        save_str = 'saves/projected_images/' + file.split('/')[-1].split('.')[0]
        os.makedirs('saves/projected_images/', exist_ok=True)
        print('Saving {}'.format(save_str + '_p.png'))
        save_image(generated, save_str + '_p.png', normalize=True)
        torch.save(latents.detach().cpu(), save_str + '.pt')
