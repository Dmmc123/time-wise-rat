from dataclasses import dataclass, field


@dataclass
class DataConfig:
    window_length: int = 32
    patch_length: int = 16
    csv_dir: str = "data/raw"
    tensor_dir: str = "data/cache"
    non_stat_datasets: list[str] = field(default_factory=lambda: ["sp_500"])
