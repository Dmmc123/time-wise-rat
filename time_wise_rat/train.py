from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.core.config_store import ConfigStore
from time_wise_rat.data import DataManager
from time_wise_rat.augmentations import (
    BaselineDataModule
)
from time_wise_rat.models import (
    PatchTST,
    AutoFormer
)
from time_wise_rat.configs import (
    ExperimentConfig
)
from typing import Mapping
from pathlib import Path

import pytorch_lightning as pl
import hydra


def train(exp_cfg: ExperimentConfig) -> Mapping[str, float]:
    # load dataloaders and ra callback
    data_manager = DataManager(cfg=exp_cfg)
    aug_data_class_name = {
        "baseline": BaselineDataModule
    }[exp_cfg.aug.aug_name]
    data_module = aug_data_class_name(
        cfg=exp_cfg,
        data_manager=data_manager
    )
    # create (or load from existing) model
    model_class = {
        "patchtst": PatchTST,
        "autoformer": AutoFormer
    }
    model = model_class[exp_cfg.model.model_name](cfg=exp_cfg)
    # create training callbacks
    weights_dir = Path(exp_cfg.train.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = weights_dir / exp_cfg.model.model_name / exp_cfg.data.dataset_name / exp_cfg.aug.aug_name
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_rmse",
        save_top_k=3,
        filename="{epoch}-{val_rmse:.4f}"
    )
    logs_dir = Path(exp_cfg.train.logs_dir) / exp_cfg.model.model_name / exp_cfg.data.dataset_name / exp_cfg.aug.aug_name
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
        datamodule=data_module
    )
    # evaluate the model
    metrics = trainer.test(dataloaders=data_module, ckpt_path="best")
    return metrics[0]


cs = ConfigStore.instance()
cs.store(name="exp", node=ExperimentConfig)


@hydra.main(version_base=None, config_name="exp")
def run_experiments(cfg: ExperimentConfig) -> None:
    train(exp_cfg=cfg)


if __name__ == "__main__":
    run_experiments()
