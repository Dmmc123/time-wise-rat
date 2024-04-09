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


class MOEDecomposition(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.model.n_experts,
            kernel_size=cfg.model.ma_length,
            padding=cfg.model.ma_length//2,
            stride=1
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # deal with original time series
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # (B, N) -> (B, 1, N)
            x = self.moe(x)
            x = x.mean(dim=1)   # (B, N_c, N) -> (B, N)
            return x
        # deal with a context
        bs, cnt, d = x.size()
        x = x.view(bs*cnt, d)  # (B, C, N) -> (B*C, N)
        x = x.unsqueeze(1)     # (B*C, N) -> (B*C, 1, N)
        x = self.moe(x)
        x = x.mean(dim=1)      # (B*C, N_c, N) -> (B, N)
        x = x.view(bs, cnt, d)
        return x


class FrequencyEnhancedBlock(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear_q = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_k = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_v = nn.Linear(cfg.data.window_length, cfg.data.patch_length)
        self.linear_out = nn.Linear(cfg.data.patch_length, cfg.data.window_length)

    def forward(self, x: Tensor) -> Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Apply the Fourier transform
        k_fft = torch.fft.fft(k, dim=-1)
        q_fft = torch.fft.fft(q, dim=-1)
        v_fft = torch.fft.fft(v, dim=-1)

        qk_fft = k_fft * torch.conj(q_fft)
        attention_scores_fft = qk_fft / torch.sqrt(torch.tensor(k.size(-1)))

        weighted_v_fft = attention_scores_fft * v_fft
        weighted_v = torch.fft.ifft(weighted_v_fft, dim=-1)

        out = self.linear_out(weighted_v.real)
        return out


class FEDformerEncoder(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe_blocks = nn.ModuleList([
            MOEDecomposition(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.fe_blocks = nn.ModuleList([
            FrequencyEnhancedBlock(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for moe_block, fe_block in zip(self.moe_blocks, self.fe_blocks):
            x = moe_block(x)
            x = fe_block(x)
        return x


class FEDformerDecoder(nn.Module):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.moe_blocks = nn.ModuleList([
            MOEDecomposition(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.fe_fr_blocks = nn.ModuleList([
            FrequencyEnhancedBlock(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.fe_sn_blocks = nn.ModuleList([
            FrequencyEnhancedBlock(cfg=cfg) for _ in range(cfg.model.num_layers)
        ])
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.data.window_length, 1)
        )

    def forward(self, x: Tensor, cnt: Optional[Tensor] = None) -> Tensor:
        if cnt is None:
            cnt = x.clone().unsqueeze(1)
        for fe_1_block, fe_2_block, moe_block in zip(
                self.fe_fr_blocks,
                self.fe_sn_blocks,
                self.moe_blocks
        ):
            x_1 = fe_1_block(x)
            x_2 = fe_2_block(cnt).mean(dim=1)
            x = x_1 + x_2
            x = moe_block(x)
        return self.head(x)


class FEDformer(pl.LightningModule, BaselineModel):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FEDformerEncoder(cfg=cfg)
        self.decoder = FEDformerDecoder(cfg=cfg)

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
