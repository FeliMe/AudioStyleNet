import argparse
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_models import models, model_utils, vgg_face
from utils import datasets

HOME = os.path.expanduser('~')


class AFFSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(AFFSystem, self).__init__()

        self.hparams = hparams

        self.model = models.VGG_Face_VA()

    def forward(self, x):
        x = self.model(x)
        return x

    def _accuracy(self, y_pred, y):
        y_pred = (((y_pred + 1) / 2).round() - 0.5) * 2
        y = (((y + 1) / 2).round() - 0.5) * 2
        acc_valence = (y_pred[:, 0] == y[:, 0]).sum().float() / y.size(0)
        acc_arousal = (y_pred[:, 1] == y[:, 1]).sum().float() / y.size(0)

        # # Only valence
        # acc_valence = (y_pred[:, 0] == y[:, 0]).sum().float() / y.size(0)
        # acc_arousal = torch.tensor(0.)

        # # Only arousal
        # acc_valence = torch.tensor(0.)
        # acc_arousal = (y_pred[:, 0] == y[:, 1]).sum().float() / y.size(0)

        return acc_valence, acc_arousal

    def _loss(self, y_pred, y):
        # Only valence
        # mse_valence = F.mse_loss(y_pred, y[:, 0].unsqueeze(1))
        # mse_arousal = torch.tensor(0.)
        # loss = mse_valence

        # Only arousal
        # mse_arousal = F.mse_loss(y_pred, y[:, 1].unsqueeze(1))
        # mse_valence = torch.tensor(0.)
        # loss = mse_arousal

        mse = F.mse_loss(y_pred, y, reduction='none').mean(dim=0)
        mse_valence = mse[0]
        mse_arousal = mse[1]
        loss = 0.5 * (mse_valence + mse_arousal)
        return loss, mse_valence, mse_arousal

    def _tensorboard_log(self, mse_valence, mse_arousal, acc_valence, acc_arousal, mode):
        return {
            f'mse_valence/{mode}': mse_valence,
            f'mse_arousal/{mode}': mse_arousal,
            f'acc_valence/{mode}': acc_valence,
            f'acc_arousal/{mode}': acc_arousal,
        }

    def training_step(self, batch, batch_idx):
        # REQUIRED

        # Unpack batch
        x = batch['x']
        y = batch['emotion']

        # Forward
        y_pred = self.forward(x)

        # Compute loss
        loss, mse_valence, mse_arousal = self._loss(y_pred, y)

        # Accuracy
        acc_valence, acc_arousal = self._accuracy(y_pred, y)

        # Logging
        tensorboard_logs = self._tensorboard_log(
            mse_valence, mse_arousal, acc_valence, acc_arousal, mode='train')

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x = batch['x']
        y = batch['emotion']

        # Forward
        y_pred = self.forward(x)

        # Compute loss
        loss, mse_valence, mse_arousal = self._loss(y_pred, y)

        # Accuracy
        acc_valence, acc_arousal = self._accuracy(y_pred, y)

        # Logging
        ret = {
            'loss': loss,
            'mse_valence': mse_valence,
            'mse_arousal': mse_arousal,
            'acc_valence': acc_valence,
            'acc_arousal': acc_arousal
        }
        return ret

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mse_valence = torch.stack([x['mse_valence'] for x in outputs]).mean()
        avg_mse_arousal = torch.stack([x['mse_arousal'] for x in outputs]).mean()
        avg_acc_valence = torch.stack([x['acc_valence'] for x in outputs]).mean()
        avg_acc_arousal = torch.stack([x['acc_arousal'] for x in outputs]).mean()

        tensorboard_log = self._tensorboard_log(
            avg_mse_valence, avg_mse_arousal, avg_acc_valence, avg_acc_arousal, mode='val')

        return {'val_loss': avg_loss, 'log': tensorboard_log}

    # def test_step(self, batch, batch_idx):
    #     # Unpack batch
    #     x = batch['x']
    #     y = batch['emotion']

    #     # Forward
    #     y_pred = self.forward(x)

    #     # Compute loss
    #     loss, mse_valence, mse_arousal = self._loss(y_pred, y)

    #     # Accuracy
    #     acc_valence, acc_arousal = self._accuracy(y_pred, y)

    #     # Logging
    #     ret = {
    #         'loss': loss,
    #         'mse_valence': mse_valence,
    #         'mse_arousal': mse_arousal,
    #         'acc_valence': acc_valence,
    #         'acc_arousal': acc_arousal
    #     }
    #     return ret

    # def test_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     avg_mse_valence = torch.stack(
    #         [x['mse_valence'] for x in outputs]).mean()
    #     avg_mse_arousal = torch.stack(
    #         [x['mse_arousal'] for x in outputs]).mean()
    #     avg_acc_valence = torch.stack(
    #         [x['acc_valence'] for x in outputs]).mean()
    #     avg_acc_arousal = torch.stack(
    #         [x['acc_arousal'] for x in outputs]).mean()

    #     tensorboard_log = self._tensorboard_log(
    #         avg_mse_valence, avg_mse_arousal, avg_acc_valence, avg_acc_arousal, mode='test')

    #     return {'test_loss': avg_loss, 'log': tensorboard_log}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def __dataloader(self, mode):
        # init data generators
        paths, annotations = datasets.aff_wild_get_paths(
            self.hparams.data_path + f'{mode}/',
            flat=False
        )
        ds = datasets.AffWild2Dataset(
            paths,
            annotations,
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            image_size=256,
        )
        data_loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams.batch_size, drop_last=True,
            shuffle=True, pin_memory=True)
        return data_loader

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.__dataloader(mode='train')

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.__dataloader(mode='validation')

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return self.__dataloader(mode='test')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=HOME + '/Datasets/aff_wild2/Aligned256/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--log_dir', type=str,
                        default='tensorboard_logs/train_omg_binary/')
    hparams = parser.parse_args()

    system = AFFSystem(hparams)

    # tb_logger = pl.logging.TensorBoardLogger(hparams.log_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath='lightning_logs/',
        save_top_k=-1,
        period=10,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    trainer = pl.Trainer(gpus=1,
                         show_progress_bar=True,
                         default_save_path='lightning_logs/',
                         weights_save_path='lightning_logs/',
                        #  checkpoint_callback=checkpoint_callback,
                        #  val_check_interval=0.01,
                        #  val_percent_check=0.01,
                         min_epochs=100)

    trainer.fit(system)
