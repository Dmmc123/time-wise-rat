from time_wise_rat.configs import ExperimentConfig
from time_wise_rat.models import BaselineModel
from torch import nn, Tensor
from typing import Optional


import pytorch_lightning as pl
import torch
import math


class Stationarizer(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True) + self.eps
        return (x - mu) / sigma, mu, sigma

    def denormalize(self, x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return x * sigma + mu


class DestationaryAttention(nn.Module):

    def __init__(
            self,
            input_size: int,
            proj_dim: int,
            narrow_out_proj: bool = False
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.input_size = input_size
        self.proj_q = nn.Linear(input_size, proj_dim)
        self.proj_k = nn.Linear(input_size, proj_dim)
        self.proj_v = nn.Linear(input_size, proj_dim)
        out_size = input_size if not narrow_out_proj else input_size // 2
        self.proj_out = nn.Linear(proj_dim, out_size)
        self.scale = math.sqrt(proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        attn = torch.bmm(q.unsqueeze(2), k.unsqueeze(1))
        attn /= self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.unsqueeze(-1)).squeeze(-1)
        out = self.proj_out(out)

        return out


class NSTransformer(pl.LightningModule, BaselineModel):

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stationarizer = Stationarizer()
        self.encoder = nn.ModuleList(
            [
                DestationaryAttention(
                    input_size=cfg.data.window_length,
                    proj_dim=cfg.model.dim_fc
                ) for _ in range(cfg.model.num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DestationaryAttention(
                    input_size=cfg.data.window_length*2,
                    proj_dim=cfg.model.dim_fc,
                    narrow_out_proj=True
                ) for _ in range(cfg.model.num_layers)
            ]
        )
        self.proj_head = nn.Linear(cfg.data.window_length, 1)

    def encode(self, x: Tensor) -> Tensor:
        x, mu, sigma = self.stationarizer(x)
        for layer in self.encoder:
            x = layer(x) + x
        x = self.stationarizer.denormalize(x, mu, sigma)
        return x

    def decode(self, x: Tensor, cnt: Optional[Tensor] = None) -> Tensor:
        if cnt is None:
            cnt = x.clone()
        else:
            b, n, l = cnt.size()
            cnt = cnt.view((b*n), l)
            cnt = self.encode(cnt)
            cnt = cnt.view(b, n, l)
            cnt = cnt.mean(dim=1)

        x, mu, sigma = self.stationarizer(x)
        x_cnt, _, _ = self.stationarizer(cnt)

        for layer in self.decoder:
            layer_input = torch.cat([x, x_cnt], dim=-1)
            x = layer(layer_input) + x
        x = self.stationarizer.denormalize(x, mu, sigma)

        return x

    def forward(self, x: Tensor, cnt: Optional[Tensor] = None) -> Tensor:
        emb = self.encode(x)
        y = self.decode(emb, cnt)
        return y
