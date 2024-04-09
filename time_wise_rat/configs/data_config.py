from dataclasses import dataclass, field
from typing import Union


@dataclass
class DataConfig:
    name: str = "btc"
    window_length: int = 32
    patch_length: int = 16
    csv_dir: str = "data/raw"
    tensor_dir: str = "data/cache"
    non_stat_datasets: list[str] = field(
        default_factory=lambda: ["sp_500"]
    )
    n_samples: Union[float, int] = 0.2
    train_size: float = 0.7
    val_size: float = 0.1
