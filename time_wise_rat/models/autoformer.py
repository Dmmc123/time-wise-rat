from time_wise_rat.configs import ExperimentConfig
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error
)
from time_wise_rat.models import BaselineModel
from torch import Tensor, nn
from typing import Optional

import pytorch_lightning as pl
import torch


class SeriesDecomposition(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.avg_pool = nn.AvgPool1d(
            kernel_size=cfg.model.ma_length,
            padding=cfg.model.ma_length//2,
            stride=1
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_trend = self.avg_pool(x)
        x_seasonal = x - x_trend
        return x_trend, x_seasonal


class AutoCorrelation(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear_q = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_k = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_v = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_out = nn.Linear(cfg.data.patch_length*2, cfg.data.window_length)

    def forward(self, x: Tensor) -> Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        k_fft = torch.fft.fft(k, dim=-1)
        q_fft = torch.fft.fft(q, dim=-1)
        qk_fft = k_fft * torch.conj(q_fft)
        qk_ifft = torch.fft.ifft(qk_fft, dim=-1)

        qk_concat = torch.cat([qk_ifft.real, v], dim=-1)
        out = self.linear_out(qk_concat)

        return out


class AutoFormerEncoder(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decomp_blocks = nn.ModuleList([
            SeriesDecomposition(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.autocor_blocks = nn.ModuleList([
            AutoCorrelation(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for autocor_block, decomp_block in zip(self.autocor_blocks, self.decomp_blocks):
            x = autocor_block(x)
            x, _ = decomp_block(x)
        return x


class AutoFormerDecoder(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decomp_blocks = nn.ModuleList([
            SeriesDecomposition(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.autocor_fr_blocks = nn.ModuleList([
            AutoCorrelation(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.autocor_sn_blocks = nn.ModuleList([
            AutoCorrelation(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.data.window_length, 1)
        )

    def forward(self, x: Tensor, cnt: Optional[Tensor] = None) -> Tensor:
        if cnt is None:
            cnt = x.clone().unsqueeze(1)
        for ac_1_block, ac_2_block, decomp_block in zip(
                self.autocor_fr_blocks,
                self.autocor_sn_blocks,
                self.decomp_blocks
        ):
            x_1 = ac_1_block(x)
            x_2 = ac_2_block(cnt).mean(dim=1)
            x = x_1 + x_2
            x, _ = decomp_block(x)
        return self.head(x)


class AutoFormer(pl.LightningModule, BaselineModel):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoFormerEncoder(cfg=cfg)
        self.decoder = AutoFormerDecoder(cfg=cfg)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x_emb: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        return self.decoder(x_emb, x_cnt)

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
