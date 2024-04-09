from time_wise_rat.configs import ExperimentConfig
from time_wise_rat.models import BaselineModel
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error
)
from torch import nn, Tensor
from typing import Optional

import pytorch_lightning as pl
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 4_096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x += self.pe[:x.size(1)].transpose(0, 1)
        return x


class PatchTST(pl.LightningModule, BaselineModel):

    def __init__(
            self, cfg: ExperimentConfig) -> None:
        # init model and save hp
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        # model layers
        self.pos_enc = PositionalEncoding(
            d_model=cfg.data.patch_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.data.patch_length,
            nhead=1,  # single head self-attention
            dim_feedforward=cfg.model.dim_fc // 2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.model.num_layers // 2,
            enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.data.patch_length,
            nhead=1,  # single head self-attention
            dim_feedforward=cfg.model.dim_fc // 2,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=cfg.model.num_layers // 2
        )
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.data.window_length * cfg.data.patch_length, 1)
        )

    def encode(self, x: Tensor) -> Tensor:
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x

    def decode(self, x_emb: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        if x_cnt is None:  # pretraining
            x_cnt = x_emb.clone()
        else:
            b, c, n, l = x_cnt.size()
            x_cnt = x_cnt.view(b, (c*n), l)
        x = self.decoder(x_emb, x_cnt)
        x = x.view(-1, self.cfg.data.window_length * self.cfg.data.patch_length)
        y = self.head(x)
        return y

    def forward(self, x: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        x_emb = self.encode(x)
        y = self.decode(x_emb, x_cnt)
        return y

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.cfg.train.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.cfg.train.scheduler_patience,
            factor=self.cfg.train.scheduling_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_rmse"
        }

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        src, tgt, cnt = batch
        pred = self(src, cnt).view(-1)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        self.log("train_mse", loss.item(), sync_dist=True, batch_size=src.size(0))
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, float]:
        src, tgt, cnt = batch
        pred = self(src, cnt).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        self.log("val_rmse", rmse, sync_dist=True, batch_size=src.size(0))
        self.log("val_mae", mae, sync_dist=True, batch_size=src.size(0))
        return {"val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, float]:
        src, tgt, cnt = batch
        pred = self(src, cnt).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        metric_dict = {"test_rmse": rmse, "test_mae": mae}
        self.log_dict(metric_dict, batch_size=src.size(0))
        return metric_dict
