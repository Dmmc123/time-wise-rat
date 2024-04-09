from time_wise_rat.data.datasets import DataManager
from time_wise_rat.configs import ExperimentConfig

import pytorch_lightning as pl


class BaselineDataModule(pl.LightningDataModule):

    def __init__(self, cfg: ExperimentConfig, data_manager: DataManager) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_manager = data_manager
        # empty attributes
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def setup(self, stage: str) -> None:
        self.train_ds, self.val_ds, self.test_ds = self.data_manager.get_datasets()
        self.train_dl, self.val_dl, self.test_dl = self.data_manager.get_dataloaders(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            test_ds=self.test_ds
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
