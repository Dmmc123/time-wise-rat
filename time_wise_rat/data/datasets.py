from torch.utils.data import Dataset, DataLoader
from time_wise_rat.configs import (
    DataConfig,
    ModelConfig,
    TrainConfig
)
from safetensors.torch import safe_open
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from torch import Tensor

import torch


@dataclass
class DatasetTS(Dataset):
    samples: Tensor
    targets: Tensor
    nn_emb: Optional[Tensor] = None
    nn_idx: Optional[Tensor] = None

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        src = self.samples[idx]
        tgt = self.targets[idx]
        cnt = None
        if self.nn_emb is not None and self.nn_emb is not None:
            cnt = self.nn_emb[self.nn_idx[idx]]
            n_c, n, d = cnt.size()
            cnt = cnt.view((n_c*n), d)  # concat neighbors into one long example
        return src, tgt, cnt


@dataclass
class DataManager:
    data_cfg: DataConfig
    model_cfg: ModelConfig
    train_cfg: TrainConfig

    def get_datasets(self) -> tuple[DatasetTS, DatasetTS, DatasetTS]:
        # get keys for tensor values in tensor cache
        src_key, tgt_key = {
            "patchtst": ("patches", "patch_targets")
        }[self.model_cfg.model_name]
        # read tensors
        tensor_filename = f"{self.data_cfg.dataset_name}.safetensors"
        tensor_full_path = Path(self.data_cfg.tensor_dir) / tensor_filename
        tensors = {}
        with safe_open(tensor_full_path, framework="pt", device="cpu") as f:
            for key in (src_key, tgt_key):
                tensors[key] = f.get_tensor(key)
        # compute borders for train/val/test samples
        n_train = int(self.data_cfg.train_size * tensors[src_key].size(0))
        n_val = int(self.data_cfg.val_size * tensors[src_key].size(0))
        # create datasets from samples of appropriate ranges
        train_ds = DatasetTS(
            samples=tensors[src_key][:n_train],
            targets=tensors[tgt_key][:n_train]
        )
        val_ds = DatasetTS(
            samples=tensors[src_key][n_train:n_train+n_val],
            targets=tensors[tgt_key][n_train:n_train+n_val]
        )
        test_ds = DatasetTS(
            samples=tensors[src_key][n_train+n_val:],
            targets=tensors[tgt_key][n_train+n_val:]
        )
        return train_ds, val_ds, test_ds

    def get_dataloaders(
            self,
            train_ds: DatasetTS,
            val_ds: DatasetTS,
            test_ds: DatasetTS
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        def collate_fn(batch):
            src = torch.stack([b[0] for b in batch])
            tgt = torch.stack([b[1] for b in batch])
            cnt = None
            if batch[0][2] is not None:
                torch.stack([b[2] for b in batch])
            return src, tgt, cnt
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn,
            shuffle=True
        )
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn
        )
        test_dl = DataLoader(
            dataset=test_ds,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn
        )
        return train_dl, val_dl, test_dl
