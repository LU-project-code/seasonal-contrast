from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.eurosat_datamodule import EurosatDataModule
from models.moco2_module import MocoV2

import numpy as np


class Classifier(LightningModule):

    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.encoder = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits, feats

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters())
        max_epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--backbone_type', type=str, default='imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    # argument values 
    args.gpus = 0
    args.data_dir = 'datasets/eurosat'
    args.backbone_type = 'pretrain'
    args.ckpt_path = 'pretrained-models/seco_resnet18_1m.ckpt'

    datamodule = EurosatDataModule(args.data_dir)

    if args.backbone_type == 'random':
        backbone = resnet.resnet18(pretrained=False)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'imagenet':
        backbone = resnet.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'pretrain':
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    else:
        raise ValueError()

    model = Classifier(backbone, in_features=512, num_classes=datamodule.num_classes)
    model.example_input_array = torch.zeros((1, 3, 64, 64))

    experiment_name = args.backbone_type
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    trainer = Trainer(logger=logger, max_epochs=20)
    trainer.fit(model, datamodule=datamodule)

    outputs = trainer.predict(model, datamodule=datamodule)
    logits = np.zeros([16200, 10])
    feats = np.zeros([16200, 512])
    cls = np.zeros(16200)
    last_idx = 0
    for i in outputs:
        n_samples = i[0].shape[0]
        logits[last_idx:(last_idx + n_samples), :] = i[0]
        feats[last_idx:(last_idx + n_samples), :] = i[1]

        last_idx += n_samples

    np.save('x_train.npy', feats)
    np.save('x_logits.npy', logits)

    for i, s in enumerate(datamodule.predict_dataset.samples):
        cls_name = s.stem.split('_')[0]
        cls[i] = datamodule.predict_dataset.class_to_idx[cls_name]

    np.save('cls.npy', cls)
