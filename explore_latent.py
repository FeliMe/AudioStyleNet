import glob
import os
import sys
import torch

from tqdm import tqdm
from projector import Projector
from my_models.generators import StyleGAN2Decoder
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    g = StyleGAN2Decoder().to(device).train()
    for param in g.parameters():
        param.requires_grad = False

    proj = Projector()
    proj.set_network(g)

    # Load target image
    path = sys.argv[1]
    image_files = glob.glob(path + '*.png')
    image_files += glob.glob(path + '*.jpg')

    for i, file in tqdm(enumerate(sorted(image_files))):
        print('Projecting {}'.format(file))

        # Load image
        target_image = Image.open(file)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        target_image = transform(target_image).to(device)

        # Run projector
        generated = proj.run(target_image)

        # Save results
        save_str = 'saves/explore_latent/' + file.split('/')[-1]
        os.makedirs('saves/explore_latent/', exist_ok=True)
        print('Saving {}'.format(save_str))
        save_image(generated, save_str, normalize=True)
        break
