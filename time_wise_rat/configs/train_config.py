from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 16_392
    num_workers: int = 4
    learning_rate: float = 5e-3  # 5e-3 for baseline, 3e-4 for baseline
    scheduler_patience: int = 5  # 5 for baseline, 10-20 for aug
    scheduling_factor: float = 0.5
    early_stopping_patience: int = 10  # 10 for baseline, 20-40 for aug
    logs_dir: str = "runs"
    weights_dir: str = "weights"
    epochs: int = 1_000
    min_delta: float = 0.02  # 0.02 for baseline, 0.01-0.005 for aug
