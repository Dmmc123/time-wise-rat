from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "patchtst"
    dim_fc: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    ma_length: int = 7
