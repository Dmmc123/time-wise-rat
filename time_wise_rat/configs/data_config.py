from dataclasses import dataclass, field
from typing import Union


@dataclass
class DataConfig:
    dataset_name: str
    window_length: int = 32
    patch_length: int = 16
    csv_dir: str = "data/raw"
    tensor_dir: str = "data/cache"
    non_stat_datasets: list[str] = field(
        default_factory=lambda: ["sp_500"]
    )
    n_samples: Union[float, int] = 2 ** 16
    train_size: float = 0.7
    val_size: float = 0.1