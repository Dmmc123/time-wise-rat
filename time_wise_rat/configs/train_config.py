from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 1024
