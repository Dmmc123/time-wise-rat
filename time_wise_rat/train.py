import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from time_wise_rat.datasets import (
    split_tensors_into_datasets,
    split_tensors_into_ra_datasets,
    split_tensors_into_tgt_datasets
)
from time_wise_rat.models import (
    Baseline,
    FullTransformer,
    TemplateAugmentedTransformer,
    TargetAugmentedTransformer,
    PositionalEncoding
)
from pytorch_lightning.loggers import TensorBoardLogger
from time_wise_rat.config import RatConfig
from torch.utils.data import DataLoader
from typing import Mapping
from pathlib import Path


def train(
        model: pl.LightningModule,
        config: RatConfig,
        weights_dir: Path,
        logs_dir: Path,
        dataset_name: str,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
) -> Mapping[str, float]:
    # create callbacks
    checkpoint = ModelCheckpoint(
        dirpath=weights_dir / model_name / dataset_name,
        monitor="val_rmse",
        save_top_k=3,
        filename="{epoch}-{val_rmse:.4f}"
    )
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name=f"{model_name}/{dataset_name}"
    )
    early_stopping = EarlyStopping(
        monitor="val_rmse",
        patience=config.patience
    )
    # create trainer object
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint, early_stopping],
        logger=tb_logger,
        enable_progress_bar=False,
        log_every_n_steps=1
    )
    # fit the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    # evaluate the model
    metrics = trainer.test(dataloaders=test_loader)
    return metrics[0]


def train_baseline(
        cache_dir: Path,
        dataset_name: str,
        weights_dir: Path,
        logs_dir: Path,
        config: RatConfig
) -> Mapping[str, float]:
    # construct data loaders
    train_ds, val_ds, test_ds = split_tensors_into_datasets(
        cache_dir=cache_dir,
        name=dataset_name,
        config=config
    )
    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        test_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    # initialize the model
    model = FullTransformer(config=config)
    # train the model
    return train(
        model=model,
        config=config,
        weights_dir=weights_dir,
        logs_dir=logs_dir,
        dataset_name=dataset_name,
        model_name="FullTransformer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )


def train_retrieval_augmented_baseline(
        cache_dir: Path,
        dataset_name: str,
        weights_dir: Path,
        logs_dir: Path,
        config: RatConfig
) -> Mapping[str, float]:
    # construct data loaders
    train_ds, val_ds, test_ds = split_tensors_into_ra_datasets(
        cache_dir=cache_dir,
        name=dataset_name,
        config=config
    )
    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        test_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    # initialize the model
    ckpt_files = (weights_dir / "FullTransformer" / dataset_name).glob("*.ckpt")
    best_baseline_ckpt = min(ckpt_files, key=lambda p: float(p.stem.split("=")[-1]))
    model = FullTransformer.load_from_checkpoint(
        checkpoint_path=best_baseline_ckpt,
        config=config,
        strict=False
    )
    model.pos_enc = PositionalEncoding(
        d_model=config.patch_length,
        drop_out=config.pos_enc_drop_out,
        max_len=config.num_patches*config.context_len
    )
    # train the model
    return train(
        model=model,
        config=config,
        weights_dir=weights_dir,
        logs_dir=logs_dir,
        dataset_name=dataset_name,
        model_name="RetrivalAugmentedFT",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )


def train_knn_transformer(
        cache_dir: Path,
        dataset_name: str,
        weights_dir: Path,
        logs_dir: Path,
        config: RatConfig
) -> Mapping[str, float]:
    # construct data loaders
    train_ds, val_ds, test_ds = split_tensors_into_tgt_datasets(
        cache_dir=cache_dir,
        name=dataset_name,
        config=config
    )
    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        test_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size
    )
    # initialize the model
    ckpt_files = (weights_dir / "Baseline" / dataset_name).glob("*.ckpt")
    best_baseline_ckpt = min(ckpt_files, key=lambda p: float(p.stem.split("=")[-1]))
    baseline = Baseline.load_from_checkpoint(
        checkpoint_path=best_baseline_ckpt,
        config=config
    )
    model = TargetAugmentedTransformer(baseline=baseline)
    # train the model
    return train(
        model=model,
        config=config,
        weights_dir=weights_dir,
        logs_dir=logs_dir,
        dataset_name=dataset_name,
        model_name="KNNTransformer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )


if __name__ == "__main__":
    train_retrieval_augmented_baseline(
        cache_dir=Path("data/cache"),
        dataset_name="btc",
        weights_dir=Path("weights"),
        logs_dir=Path("runs"),
        config=RatConfig()
    )
