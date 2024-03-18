from dataclasses import dataclass


@dataclass
class RatConfig:
    num_patches: int = 64
    patch_length: int = 32
    dim_fc: int = 32
    num_layers: int = 3
    pos_enc_drop_out: float = 0.1
    reg_head_drop_out: float = 0.2
    num_templates: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 4_096
    epochs: int = 10_000
    patience: int = 1000
    num_workers: int = 4
    train_size: float = 0.7
    val_size: float = 0.1
