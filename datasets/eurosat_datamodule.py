from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from datasets.eurosat_dataset import EurosatDataset
from datasets.eurosat_predict_dataset import EurosatPredictDataset


class EurosatDataModule(LightningDataModule):

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    @property
    def num_classes(self):
        return 10

    def setup(self, stage=None):
        self.train_dataset = EurosatDataset(self.data_dir, split='train', transform=T.ToTensor())
        self.val_dataset = EurosatDataset(self.data_dir, split='val', transform=T.ToTensor())
        self.predict_dataset = EurosatPredictDataset(self.data_dir, split='train', transform=T.ToTensor())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            drop_last=True,
            pin_memory=True
        )