from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.core.config_store import ConfigStore
from time_wise_rat.data import DataManager
from time_wise_rat.augmentations import (
    BaselineDataModule,
    TERADataModule,
    MQRetNNDataModule,
    RATSFDataModule
)
from time_wise_rat.models import (
    PatchTST,
    AutoFormer,
    NSTransformer,
    FEDformer
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
    aug_data_class_name, use_pretrain = {
        "baseline": (BaselineDataModule, False),
        "tera": (TERADataModule, True),
        "mqretnn": (MQRetNNDataModule, True),
        "ratsf": (RATSFDataModule, False)
    }[exp_cfg.aug.name]
    data_module = aug_data_class_name(
        cfg=exp_cfg,
        data_manager=data_manager
    )
    # create (or load from existing) model
    model_class = {
        "patchtst": PatchTST,
        "autoformer": AutoFormer,
        "nstransformer": NSTransformer,
        "fedformer": FEDformer
    }
    model = model_class[exp_cfg.model.name](cfg=exp_cfg)
    # modify hp for stable augmentation-based training
    if exp_cfg.aug.name in {"mqretnn", "ratsf"}:
        exp_cfg.train.learning_rate = 5e-5
        exp_cfg.train.scheduler_patience = 10
        exp_cfg.train.early_stopping_patience = 20
        exp_cfg.train.min_delta = 0.01
    elif exp_cfg.aug.name in {"baseline", "tera"}:
        exp_cfg.train.learning_rate = 3e-4
        exp_cfg.train.scheduler_patience = 50
        exp_cfg.train.early_stopping_patience = 100
        exp_cfg.train.min_delta = 0.02
    # if needed for augmentation - load the best model
    weights_dir = Path(exp_cfg.train.weights_dir)
    baseline_ckpt_dir = (weights_dir /
                         "baseline" /
                         exp_cfg.model.name /
                         exp_cfg.data.name)
    if use_pretrain:
        model_paths = baseline_ckpt_dir.glob("*.ckpt")
        best_path = min(model_paths, key=lambda p: float(p.stem.split("=")[-1]))
        model = model_class[exp_cfg.model.name].load_from_checkpoint(
            best_path,
            cfg=exp_cfg
        )
    # create training callbacks
    ckpt_dir = (weights_dir /
                exp_cfg.aug.name /
                exp_cfg.model.name /
                exp_cfg.data.name)
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_rmse",
        save_top_k=3,
        filename="{epoch}-{val_rmse:.4f}"
    )
    logs_dir = (Path(exp_cfg.train.logs_dir) /
                exp_cfg.aug.name /
                exp_cfg.model.name /
                exp_cfg.data.name)
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="logs"
    )
    early_stopping = EarlyStopping(
        monitor="val_rmse",
        patience=exp_cfg.train.early_stopping_patience,
        min_delta=exp_cfg.train.min_delta
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
    metrics = trainer.test(dataloaders=data_module, ckpt_path="last")
    return metrics[0]


cs = ConfigStore.instance()
cs.store(name="exp", node=ExperimentConfig)


@hydra.main(version_base=None, config_name="exp")
def run_experiments(cfg: ExperimentConfig) -> None:
    train(exp_cfg=cfg)


if __name__ == "__main__":
    run_experiments()
