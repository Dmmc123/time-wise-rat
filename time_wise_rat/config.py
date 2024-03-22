from dataclasses import dataclass


@dataclass
class RatConfig:
    num_patches: int = 32
    context_len: int = 32
    patch_length: int = 16
    dim_fc: int = 32
    num_layers: int = 1
    pos_enc_drop_out: float = 0.1
    reg_head_drop_out: float = 0.2
    num_templates: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 16
    epochs: int = 5000
    patience: int = 400
    num_workers: int = 4
    train_size: float = 0.7
    val_size: float = 0.1
