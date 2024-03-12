import pandas as pd
import numpy as np
import torch


from time_wise_rat.utils import construct_patches
from time_wise_rat.config import RatConfig
from dataclasses import dataclass, field
from safetensors.torch import save_file
from torch.utils.data import Dataset
from safetensors import safe_open
from torch import Tensor
from pathlib import Path
from typing import Self


@dataclass
class TSProcessor:

    @staticmethod
    def _extract(csv_path: Path) -> np.ndarray:
        df = pd.read_csv(csv_path)
        values = df["Value"].values
        return values

    @staticmethod
    def _transform(values: np.ndarray, config: RatConfig) -> dict[str, Tensor]:
        # apply normalizing transformations to data
        values = np.diff(values)
        # construct patches from normalized data
        patches = construct_patches(
            array=values,
            num_patches=config.num_patches,
            patch_length=config.patch_length
        )
        # crop out the required amount of samples
        n_samples = patches.shape[0] - 1
        patches = patches[:n_samples]
        targets = values[-n_samples:]
        # convert values to tensors
        patches = torch.tensor(patches, dtype=torch.float).contiguous()
        targets = torch.tensor(targets, dtype=torch.float)
        return {
            "patches": patches,
            "targets": targets
        }

    @staticmethod
    def _load(cache_dir: Path, name: str, tensors: dict[str, Tensor]) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, cache_dir / f"{name}.safetensors")

    @staticmethod
    def run(csv_path: Path, cache_dir: Path, config: RatConfig) -> None:
        values = TSProcessor._extract(csv_path=csv_path)
        tensors = TSProcessor._transform(values=values, config=config)
        TSProcessor._load(cache_dir=cache_dir, name=csv_path.stem, tensors=tensors)


@dataclass
class TSD(Dataset):
    """Time Series Dataset"""

    name: str = field(repr=True, init=True)
    patches: Tensor = field(repr=False, init=True)
    targets: Tensor = field(repr=False, init=True)

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.patches[idx], self.targets[idx]
