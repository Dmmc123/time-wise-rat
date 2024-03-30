from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    aug_name: str
    n_neighbors: int = 24
