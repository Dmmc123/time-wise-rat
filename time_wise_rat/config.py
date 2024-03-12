from dataclasses import dataclass


@dataclass
class RatConfig:
    num_patches: int = 32
    patch_length: int = 16
    dim_fc: int = 128
    num_layers: int = 3
    pos_enc_drop_out: float = 0.2
    reg_head_drop_out: float = 0.5
    num_templates: int = 5
    learning_rate: float = 3e-4
    batch_size: int = 128
