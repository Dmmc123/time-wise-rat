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
        # values = np.diff(values)
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


def split_tensors_into_datasets(
        cache_dir: Path,
        name: str,
        train_size: float,
        val_size: float,
        min_max_scale: bool
) -> tuple[TSD, TSD, TSD]:
    # read tensors
    tensors_path = cache_dir / f"{name}.safetensors"
    tensors = {}
    with safe_open(tensors_path, framework="pt", device="cpu") as f:
        for key in ("patches", "targets"):
            tensors[key] = f.get_tensor(key)
    # compute the indices
    n_samples = tensors["patches"].size(0)
    n_train = int(n_samples * train_size)
    n_val = int(n_samples * val_size)
    # crop out the tensors
    train_p = tensors["patches"][:n_train]
    train_t = tensors["targets"][:n_train]
    val_p = tensors["patches"][n_train:n_train+n_val]
    val_t = tensors["targets"][n_train:n_train+n_val]
    test_p = tensors["patches"][n_train+n_val:]
    test_t = tensors["targets"][n_train+n_val:]
    # normalize to train min-max
    if min_max_scale:
        train_min, train_max = train_p.min(), train_p.max()
        train_p = (train_p - train_min) / (train_max - train_min)
        train_t = (train_t - train_min) / (train_max - train_min)
        val_p = (val_p - train_min) / (train_max - train_min)
        val_t = (val_t - train_min) / (train_max - train_min)
        test_p = (test_p - train_min) / (train_max - train_min)
        test_t = (test_t - train_min) / (train_max - train_min)
    # construct a dataset object
    train_ds = TSD(
        name=tensors_path.stem,
        patches=train_p,
        targets=train_t
    )
    val_ds = TSD(
        name=tensors_path.stem,
        patches=val_p,
        targets=val_t
    )
    test_ds = TSD(
        name=tensors_path.stem,
        patches=test_p,
        targets=test_t
    )
    # return the datasets
    return train_ds, val_ds, test_ds
