from time_wise_rat.data.utils import (
    construct_patches,
    construct_windows,
    get_log_return
)
from time_wise_rat.configs import DataConfig
from safetensors.torch import save_file
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor

import pandas as pd
import numpy as np
import torch


@dataclass
class TableToTensorPreprocessor:
    config: DataConfig

    def _extract(self) -> np.ndarray:
        # read csv with time series data
        dataset_path = f"{self.config.csv_dir}/{self.config.dataset_name}.csv"
        df = pd.read_csv(dataset_path)
        # return column with time series
        return df["Value"].values

    def _transform(self, data: np.ndarray) -> dict[str, Tensor]:
        # reduce stationarity of dataset of needed
        if self.config.dataset_name in self.config.non_stat_datasets:
            data = get_log_return(data)
        # normalize data according to training part
        n_train = int(data.shape[0] * self.config.train_size)
        t_min, t_max = data[:n_train].min(), data[:n_train].max()
        data = (data - t_min) / (t_max - t_min)
        # construct patches, windows and targets
        patches = construct_patches(
            array=data,
            num_patches=self.config.window_length,
            patch_length=self.config.patch_length
        )
        windows = construct_windows(
            array=data,
            window_length=self.config.window_length
        )
        # crop out the required amount of samples
        n_patch_samples = patches.shape[0] - 1
        n_window_samples = windows.shape[0] - 1
        patches = patches[:n_patch_samples]
        windows = windows[:n_window_samples]
        patch_targets = data[-n_patch_samples:]
        window_targets = data[-n_window_samples:]
        # convert arrays to tensors
        patches = torch.tensor(patches, dtype=torch.float).contiguous()
        windows = torch.tensor(windows, dtype=torch.float).contiguous()
        patch_targets = torch.tensor(patch_targets, dtype=torch.float)
        window_targets = torch.tensor(window_targets, dtype=torch.float)
        # get amount of pruned elements in datasets
        n_prune_patch_samples = None
        n_prune_window_samples = None
        if isinstance(self.config.n_samples, float):
            n_prune_patch_samples = int(self.config.n_samples * n_patch_samples)
            n_prune_window_samples = int(self.config.n_samples * n_window_samples)
        elif isinstance(self.config.n_samples, int):
            n_prune_patch_samples = min(self.config.n_samples, n_patch_samples)
            n_prune_window_samples = min(self.config.n_samples, n_window_samples)
        patches_idx = torch.randperm(n_patch_samples)[:n_prune_patch_samples]
        windows_idx = torch.randperm(n_window_samples)[:n_prune_window_samples]
        # randomly select a subset of dataset while maintaining order
        patches_idx, _ = torch.sort(patches_idx)
        windows_idx, _ = torch.sort(windows_idx)
        patches = patches[patches_idx]
        patch_targets = patch_targets[patches_idx]
        windows = windows[windows_idx]
        window_targets = window_targets[windows_idx]
        # return a named tensor collection
        return {
            "patches": patches,
            "windows": windows,
            "patch_targets": patch_targets,
            "window_targets": window_targets
        }

    def _load(self, data: dict[str, Tensor]) -> None:
        # create folder with tensors if it doesn't exist
        tensor_dir = Path(self.config.tensor_dir)
        tensor_dir.mkdir(parents=True, exist_ok=True)
        # save named tensors
        tensor_filename = f"{self.config.dataset_name}.safetensors"
        save_file(data, tensor_dir / tensor_filename)

    def run(self) -> None:
        data = self._extract()
        data = self._transform(data=data)
        self._load(data=data)
