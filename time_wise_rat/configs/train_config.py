from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 1024
    num_workers: int = 3
    learning_rate: float = 3e-4
    scheduler_patience: int = 5
    scheduling_factor: float = 0.5
    early_stopping_patience: int = 10
    logs_dir: str = "runs"
    weights_dir: str = "weights"
    epochs: int = 100
