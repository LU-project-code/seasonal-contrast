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

# Shirin imports
from PIL import Image
from torchvision import transforms as T
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
        return logits

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
        logits = self(x)
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

    # load and split the dataset into training and validation
    datamodule.setup() # load filenames for training and testing datasets
    tnsr_trnsfrm = T.ToTensor()

    # obtain image representations for training data
    x_train = np.zeros([len(datamodule.train_dataset.samples), 512])
    y_train = -1 * np.ones(len(datamodule.train_dataset.samples))
    for i, s in enumerate(datamodule.train_dataset.samples):
        # load and convert the image to torch tensor
        img = tnsr_trnsfrm(Image.open(s))
        img = torch.unsqueeze(img, 0) # add a dimension at the start for the minibatch

        x_train[i, :] = model.encoder(img).detach().numpy()

        cls_name = s.stem.split('_')[0]
        y_train[i] = datamodule.train_dataset.class_to_idx[cls_name]

        if (i % 100) == 0:
            print(f'Converted {i} images')

    # obtain image representations for training data
    x_val = np.zeros([len(datamodule.val_dataset.samples), 512])
    y_val = -1 * np.ones(len(datamodule.val_dataset.samples))
    for i, s in enumerate(datamodule.val_dataset.samples):
        # load and convert the image to torch tensor
        img = tnsr_trnsfrm(Image.open(s))
        img = torch.unsqueeze(img, 0) # add a dimension at the start for the minibatch

        x_val[i, :] = model.encoder(img).detach().numpy()

        cls_name = s.stem.split('_')[0]
        y_val[i] = datamodule.val_dataset.class_to_idx[cls_name]

        if (i % 100) == 0:
            print(f'Converted {i} images')

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)

    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)


        

    # trainer = Trainer(logger=logger, max_epochs=100)
    # trainer.fit(model, datamodule=datamodule)
