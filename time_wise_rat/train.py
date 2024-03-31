from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.core.config_store import ConfigStore
from time_wise_rat.data import DataManager
from time_wise_rat.models import PatchTST
from time_wise_rat.configs import (
    ExperimentConfig
)
from typing import Mapping
from pathlib import Path

import pytorch_lightning as pl
import hydra


def train(exp_cfg: ExperimentConfig) -> Mapping[str, float]:
    # load dataloaders
    data_manager = DataManager(cfg=exp_cfg)
    train_ds, val_ds, test_ds = data_manager.get_datasets()
    train_dl, val_dl, test_dl = data_manager.get_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds
    )
    # create (or load from existing) model
    model = PatchTST(cfg=exp_cfg)
    # create training callbacks
    weights_dir = Path(exp_cfg.train.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = weights_dir / exp_cfg.model.model_name / exp_cfg.data.dataset_name
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_rmse",
        save_top_k=3,
        filename="{epoch}-{val_rmse:.4f}"
    )
    logs_dir = Path(exp_cfg.train.logs_dir) / exp_cfg.model.model_name / exp_cfg.data.dataset_name
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="logs"
    )
    early_stopping = EarlyStopping(
        monitor="val_rmse",
        patience=exp_cfg.train.early_stopping_patience
    )
    # create trainer object
    trainer = pl.Trainer(
        max_epochs=exp_cfg.train.epochs,
        callbacks=[checkpoint, early_stopping],
        logger=tb_logger,
        enable_progress_bar=False,
        log_every_n_steps=1
    )
    # fit the model
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
    # evaluate the model
    metrics = trainer.test(dataloaders=test_dl)
    return metrics[0]


cs = ConfigStore.instance()
cs.store(name="exp", node=ExperimentConfig)


@hydra.main(version_base=None, config_name="exp")
def run_experiments(cfg: ExperimentConfig) -> None:
    train(exp_cfg=cfg)


if __name__ == "__main__":
    run_experiments()
