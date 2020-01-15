import os
import sys
import torch

from my_models.generators import StyleGAN2Decoder
from torchvision.utils import save_image

if __name__ == "__main__":
    assert len(sys.argv) == 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load latents
    latent1 = torch.load(sys.argv[1]).to(device)
    latent2 = torch.load(sys.argv[2]).to(device)

    # Load model
    G = StyleGAN2Decoder().to(device).train()
    for param in G.parameters():
        param.requires_grad = False

    os.makedirs('saves/latent_interpolation/', exist_ok=True)

    steps = 90

    for t in range(steps):
        w = latent1 + (t / steps) * (latent2 - latent1)
        img, _ = G.g([w], input_is_latent=True)

        save_image(img, 'saves/latent_interpolation/{}.png'.format(str(t + 1).zfill(3)),
                   normalize=True, range=(-1, 1))
