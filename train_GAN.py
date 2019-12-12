import argparse
import importlib
import os

import dataloader

from solver_gan import GANSolver

HOME = os.path.expanduser('~')
PLOT_GRADS = False

if PLOT_GRADS:
    print("WARNING: Plot gradients is on. This may cause slow training time!")


""" Load config """

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='ravdess_im_gan',
                    help='name of config')
args = parser.parse_args()

config = importlib.import_module('configs.' + args.config).config


""" Load dataset """

# ds = dataloader.CELEBADataset(
#     root_path=HOME + '/Datasets/CELEBA/LandmarksLineImage',
#     target_path=HOME + '/Datasets/CELEBA/Imgs',
#     normalize=config.normalize,
#     mean=config.mean,
#     std=config.std,
#     img_size=config.image_size
# )

ds = dataloader.RAVDESSDSPix2Pix(root_path=config.data_path,
                                 target_root_path=config.target_data_path,
                                 data_format=config.data_format,
                                 use_same_sentence=config.use_same_sentence,
                                 normalize=config.normalize,
                                 mean=config.mean,
                                 std=config.std,
                                 max_samples=None,
                                 seed=config.random_seed,
                                 image_size=config.image_size,
                                 label_one_hot=True,
                                 emotions=config.emotions)

1 / 0

print("Found {} samples in total".format(len(ds)))

data_loaders, dataset_sizes = dataloader.get_data_loaders(
    ds, config.validation_split, config.batch_size, config.use_cuda)

print("Using {} samples for training and {} for validation".format(
    dataset_sizes['train'], dataset_sizes['val']))


""" Show data example """

sample = next(iter(data_loaders['train']))
print('Input Shape: {}'.format(sample['A'].shape))
# ds.show_sample()


""" Initialize solver """

solver = GANSolver(config)


""" Do training """

solver.train_model(data_loaders,
                   PLOT_GRADS)
