from time_wise_rat.configs.augmentation_config import AugmentationConfig
from time_wise_rat.models import BaselineModel
from time_wise_rat.configs import (
    AugmentationConfig,
    ModelConfig,
    TrainConfig,
    DataConfig
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

    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        # init model and save hp
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.save_hyperparameters()
        # model layers
        self.pos_enc = PositionalEncoding(
            d_model=data_cfg.patch_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=data_cfg.patch_length,
            nhead=1,  # single head self-attention
            dim_feedforward=model_cfg.dim_fc,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=model_cfg.num_layers,
            enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=data_cfg.patch_length,
            nhead=1,  # single head self-attention
            dim_feedforward=model_cfg.dim_fc,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=model_cfg.num_layers
        )
        self.head = nn.Sequential(
            nn.Dropout(p=model_cfg.dropout),
            nn.Linear(data_cfg.window_length * data_cfg.patch_length, 1)
        )

    def encode(self, x: Tensor) -> Tensor:
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x

    def decode(self, x_emb: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        if x_cnt is None:  # pretraining
            x_cnt = x_emb.clone()
        x = self.decoder(x_emb, x_cnt)
        x = x.view(-1, self.data_cfg.window_length * self.data_cfg.patch_length)
        y = self.head(x)
        return y

    def forward(self, x: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        x_emb = self.encode(x)
        y = self.decode(x_emb, x_cnt)
        return y


if __name__ == "__main__":
    data_cfg = DataConfig(dataset_name="aboba")
    model_cfg = ModelConfig(model_name="saas")
    train_cfg = TrainConfig()
    aug_cfg = AugmentationConfig(aug_name="terra")
    device = torch.device("cuda")

    model = PatchTST(data_cfg=data_cfg, model_cfg=model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.rand(
        train_cfg.batch_size,
        data_cfg.window_length,
        data_cfg.patch_length
    ).to(device)
    x_cnt = torch.rand(
        train_cfg.batch_size,
        data_cfg.window_length * aug_cfg.n_neighbors,
        data_cfg.patch_length
    ).to(device)

    y_1 = model(x)
    y_2 = model(x, x_cnt)

    print(y_1.size(), y_2.size())
