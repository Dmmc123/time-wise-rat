from dataclasses import dataclass


@dataclass
class RatConfig:
    num_patches: int = 32
    patch_length: int = 16
