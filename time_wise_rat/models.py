from time_wise_rat.config import RatConfig
from torch import Tensor, nn
from sklearn.metrics import (
    mean_squared_error,
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
            nn.Linear(config.num_patches * config.patch_length, config.dim_fc),
            nn.Dropout(p=config.reg_head_drop_out),
            nn.Linear(config.dim_fc, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # encode the original time series
        x = self.pos_enc(x)
        out = self.encoder(x)
        # flat out encoder representations before regression
        out = out.view(-1, config.num_patches * config.patch_length)
        out = self.head(out)
        return out

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        src, tgt = batch
        pred = self(x=src)
        loss = torch.nn.functional.mse_loss(pred, tgt)
        self.log("train_mse", loss.item(), sync_dist=True, batch_size=src.size(0))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt = batch
        pred = self(x=src)
        rmse = mean_squared_error(tgt.numpy(), pred.numpy(), squared=False)
        mae = mean_absolute_error(tgt.numpy(), pred.numpy())
        self.log("val_rmse", rmse, sync_dist=True, batch_size=src.size(0))
        self.log("val_mae", mae, sync_dist=True, batch_size=src.size(0))
        return {"val_rmse": rmse, "val_mae": mae}

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict[str, float]:
        src, tgt = batch
        pred = self(x=src)
        rmse = mean_squared_error(tgt.numpy(), pred.numpy(), squared=False)
        mae = mean_absolute_error(tgt.numpy(), pred.numpy())
        metric_dict = {"test_rmse": rmse, "test_mae": mae}
        self.log_dict(metric_dict)
        return metric_dict


if __name__ == "__main__":
    config = RatConfig()
    model = Baseline(config=config)
    x = torch.rand(config.batch_size, config.num_patches, config.patch_length)
    y = model(x)
    print(y.size())


