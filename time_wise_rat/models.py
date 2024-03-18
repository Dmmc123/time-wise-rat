from time_wise_rat.config import RatConfig
from torch import Tensor, nn
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error
)

import torch
import math
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):

    def __init__(
            self,
            d_model: int,
            drop_out: float = 0.1,
            max_len: int = 1_024
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=drop_out)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x += self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class Baseline(pl.LightningModule):

    def __init__(self, config: RatConfig) -> None:
        super().__init__()
        self.config = config
        # time series encoder
        self.pos_enc = PositionalEncoding(
            d_model=config.patch_length,
            drop_out=config.pos_enc_drop_out,
            max_len=config.num_patches
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.patch_length,
            nhead=1,  # single head self-attention
            dim_feedforward=config.dim_fc,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False
        )
        # regression head
        self.head = nn.Sequential(
            nn.Dropout(p=config.reg_head_drop_out),
            nn.Linear(config.num_patches * config.patch_length, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # encode the original time series
        x = self.pos_enc(x)
        out = self.encoder(x)
        # flat out encoder representations before regression
        out = out.view(-1, self.config.num_patches * self.config.patch_length)
        out = self.head(out)
        return out

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        src, tgt = batch
        pred = self(x=src).view(-1)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        self.log("train_mse", loss.item(), sync_dist=True, batch_size=src.size(0))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt = batch
        pred = self(x=src).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        self.log("val_rmse", rmse, sync_dist=True, batch_size=src.size(0))
        self.log("val_mae", mae, sync_dist=True, batch_size=src.size(0))
        return {"val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt = batch
        pred = self(x=src).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        metric_dict = {"test_rmse": rmse, "test_mae": mae}
        self.log_dict(metric_dict)
        return metric_dict


class TemplateAugmentedTransformer(pl.LightningModule):

    def __init__(self, baseline: Baseline) -> None:
        super().__init__()
        self.config = baseline.config
        # time series encoder
        self.pos_enc = baseline.pos_enc
        self.encoder = baseline.encoder
        # regression head
        self.head = baseline.head
        self.templ_mat = nn.Parameter(torch.rand(
            self.config.patch_length,
            self.config.patch_length
        ))

    def forward(self, x: Tensor, x_templ: Tensor) -> Tensor:
        # encode the original time series
        x = self.pos_enc(x)
        out = self.encoder(x)
        # flat out encoder representations before regression
        out = out.view(-1, self.config.num_patches * self.config.patch_length)
        # transform the templates
        templ_out = x_templ @ self.templ_mat
        templ_out = templ_out.mean(dim=1)  # average across templates
        templ_out = templ_out.view(-1, self.config.num_patches * self.config.patch_length)
        # correct baseline features with template features
        out += templ_out
        # regress the price
        out = self.head(out)
        return out

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        src, tgt, templ = batch
        pred = self(x=src, x_templ=templ).view(-1)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        self.log("train_mse", loss.item(), sync_dist=True, batch_size=src.size(0))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt, templ = batch
        pred = self(x=src, x_templ=templ).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        self.log("val_rmse", rmse, sync_dist=True, batch_size=src.size(0))
        self.log("val_mae", mae, sync_dist=True, batch_size=src.size(0))
        return {"val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt, templ = batch
        pred = self(x=src, x_templ=templ).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        metric_dict = {"test_rmse": rmse, "test_mae": mae}
        self.log_dict(metric_dict)
        return metric_dict


class TargetAugmentedTransformer(pl.LightningModule):

    def __init__(self, baseline: Baseline) -> None:
        super().__init__()
        self.config = baseline.config
        # time series encoder
        self.pos_enc = baseline.pos_enc
        self.encoder = baseline.encoder
        # regression head
        self.head = baseline.head
        self.target_attn = nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
            batch_first=True
        )

    def forward(self, x: Tensor, y_templ: Tensor) -> Tensor:
        # encode the original time series
        out = self.pos_enc(x)
        out = self.encoder(out)
        # flat out encoder representations before regression
        out = out.view(-1, self.config.num_patches * self.config.patch_length)
        # regress the price
        out = self.head(out)
        # correct the price by attending to targets of templates
        b = x.size(0)
        out = out.view(b, 1, 1)
        y_templ = y_templ.view(b, -1, 1)
        out, _ = self.target_attn(query=out, key=y_templ, value=y_templ)
        return out

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        src, tgt, templ_tgt = batch
        pred = self(x=src, y_templ=templ_tgt).view(-1)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        self.log("train_mse", loss.item(), sync_dist=True, batch_size=src.size(0))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt, templ_tgt = batch
        pred = self(x=src, y_templ=templ_tgt).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        self.log("val_rmse", rmse, sync_dist=True, batch_size=src.size(0))
        self.log("val_mae", mae, sync_dist=True, batch_size=src.size(0))
        return {"val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt, templ_tgt = batch
        pred = self(x=src, y_templ=templ_tgt).view(-1)
        rmse = root_mean_squared_error(tgt.cpu().numpy(), pred.cpu().numpy())
        mae = mean_absolute_error(tgt.cpu().numpy(), pred.cpu().numpy())
        metric_dict = {"test_rmse": rmse, "test_mae": mae}
        self.log_dict(metric_dict)
        return metric_dict


if __name__ == "__main__":
    config = RatConfig()
    model = Baseline(config=config)
    x = torch.rand(
        config.batch_size,
        config.num_patches,
        config.patch_length
    )
    y_templ = torch.rand(
        config.batch_size,
        config.num_templates
    )
    rat = TargetAugmentedTransformer(baseline=model)
    y = rat(x, y_templ)
    print(y.size())


