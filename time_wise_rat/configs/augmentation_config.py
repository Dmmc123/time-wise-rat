from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    aug_name: str = "baseline"
