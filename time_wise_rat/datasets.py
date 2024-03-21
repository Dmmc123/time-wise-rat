import pandas as pd
import numpy as np
import faiss
import torch
import tqdm

from torch.utils.data import Dataset, DataLoader, TensorDataset
from time_wise_rat.utils import construct_patches
from time_wise_rat.config import RatConfig
from time_wise_rat.models import FullTransformer
from dataclasses import dataclass, field
from safetensors.torch import save_file
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
        n_train = int(values.shape[0] * config.train_size)
        t_min, t_max = values[:n_train].min(), values[:n_train].max()
        values = (values - t_min) / (t_max - t_min)
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
class ContextProcessor:

    @staticmethod
    def _extract(tensor_path: Path) -> dict[str, Tensor]:
        tensors = {}
        with safe_open(tensor_path, framework="pt", device="cpu") as f:
            for key in ("patches", "targets"):
                tensors[key] = f.get_tensor(key)
        return tensors

    @staticmethod
    def _transform(
            tensors: dict[str, Tensor],
            baseline_ckpt_path: Path,
            config: RatConfig
    ) -> dict[str, Tensor]:
        # make loader for training data
        n_train = int(tensors["patches"].size(0) * config.train_size)
        p_ds = TensorDataset(tensors["patches"])
        loader = DataLoader(
            dataset=p_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        # load model and do inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullTransformer.load_from_checkpoint(
            checkpoint_path=baseline_ckpt_path,
            config=config
        ).to(device)
        embeddings = []
        with torch.no_grad():
            for (batch,) in tqdm.tqdm(loader, desc="Computing embeddings"):
                b = batch.size(0)
                out = model.pos_enc(batch.to(device))
                out = model.encoder(out).to("cpu")
                out = out.view(b, -1).numpy()
                embeddings.append(out)
        embeddings = np.concatenate(embeddings, axis=0)
        train_embs = embeddings[:n_train]
        # index embeddings
        index = faiss.IndexFlatIP(train_embs.shape[1])
        index.add(train_embs)
        # retrieve nearest neighbors
        _, nn_idx = index.search(embeddings, k=config.context_len)
        nn_idx = torch.tensor(nn_idx, dtype=torch.long)
        # return enriched tensors
        tensors["nn_idx"] = nn_idx
        return tensors

    @staticmethod
    def _load(cache_dir: Path, name: str, tensors: dict[str, Tensor]) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, cache_dir / f"{name}.safetensors")

    @staticmethod
    def run(
            cache_dir: Path,
            name: str,
            baseline_ckpt_path: Path,
            config: RatConfig
    ) -> None:
        tensors = ContextProcessor._extract(
            tensor_path=cache_dir / f"{name}.safetensors"
        )
        tensors = ContextProcessor._transform(
            tensors=tensors,
            baseline_ckpt_path=baseline_ckpt_path,
            config=config
        )
        ContextProcessor._load(
            cache_dir=cache_dir,
            name=name,
            tensors=tensors
        )


@dataclass
class TGTProcessor(ContextProcessor):

    @staticmethod
    def _transform(
            tensors: dict[str, Tensor],
            baseline_ckpt_path: Path,
            config: RatConfig
    ) -> dict[str, Tensor]:
        # make loader for training data
        n_train = int(tensors["patches"].size(0) * config.train_size)
        p_ds = TensorDataset(tensors["patches"])
        loader = DataLoader(
            dataset=p_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        # load model and do inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullTransformer.load_from_checkpoint(
            checkpoint_path=baseline_ckpt_path,
            config=config
        ).to(device)
        embeddings = []
        with torch.no_grad():
            for (batch,) in tqdm.tqdm(loader, desc="Computing embeddings"):
                # out = model.pos_enc(batch.to(device))
                out = model.encoder(batch.to(device)).to("cpu")
                out = out.mean(dim=1).numpy()
                embeddings.append(out)
        embeddings = np.concatenate(embeddings, axis=0)
        train_embs = embeddings[:n_train]
        # index embeddings
        index = faiss.IndexFlatIP(train_embs.shape[1])
        index.add(train_embs)
        # retrieve nearest neighbors
        _, nn_idx = index.search(embeddings, k=config.num_templates)
        nn_idx = torch.tensor(nn_idx, dtype=torch.long)
        # return enriched tensors
        tensors["nn_tgt_idx"] = nn_idx
        return tensors

    @staticmethod
    def run(
            cache_dir: Path,
            name: str,
            baseline_ckpt_path: Path,
            config: RatConfig
    ) -> None:
        tensors = super()._extract(
            tensor_path=cache_dir / f"{name}.safetensors"
        )
        tensors = TGTProcessor._transform(
            tensors=tensors,
            baseline_ckpt_path=baseline_ckpt_path,
            config=config
        )
        super()._load(
            cache_dir=cache_dir,
            name=name,
            tensors=tensors
        )


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


@dataclass
class TemplateDataset(Dataset):
    """Retrieval Augmented Time Series Dataset"""

    name: str = field(repr=True, init=True)
    patches: Tensor = field(repr=False, init=True)
    targets: Tensor = field(repr=False, init=True)
    train_patches: Tensor = field(repr=False, init=True)
    nn_idx: Tensor = field(repr=False, init=True)

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.patches[idx],
            self.targets[idx],
            self.train_patches[self.nn_idx[idx]]
        )


@dataclass
class TargetDataset(Dataset):
    """Retrieval Augmented Time Series Dataset"""

    name: str = field(repr=True, init=True)
    patches: Tensor = field(repr=False, init=True)
    targets: Tensor = field(repr=False, init=True)
    train_targets: Tensor = field(repr=False, init=True)
    nn_tgt_idx: Tensor = field(repr=False, init=True)

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.patches[idx],
            self.targets[idx],
            self.train_targets[self.nn_tgt_idx[idx]]
        )


def split_tensors_into_datasets(
        cache_dir: Path,
        name: str,
        config: RatConfig
) -> tuple[TSD, TSD, TSD]:
    # read tensors
    tensors_path = cache_dir / f"{name}.safetensors"
    tensors = {}
    with safe_open(tensors_path, framework="pt", device="cpu") as f:
        for key in ("patches", "targets"):
            tensors[key] = f.get_tensor(key)
    # compute the indices
    n_samples = tensors["patches"].size(0)
    n_train = int(n_samples * config.train_size)
    n_val = int(n_samples * config.val_size)
    # crop out the tensors
    train_p = tensors["patches"][:n_train]
    train_t = tensors["targets"][:n_train]
    val_p = tensors["patches"][n_train:n_train+n_val]
    val_t = tensors["targets"][n_train:n_train+n_val]
    test_p = tensors["patches"][n_train+n_val:]
    test_t = tensors["targets"][n_train+n_val:]
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


def split_tensors_into_ra_datasets(
        cache_dir: Path,
        name: str,
        config: RatConfig
) -> tuple[TemplateDataset, TemplateDataset, TemplateDataset]:
    # read tensors
    tensors_path = cache_dir / f"{name}.safetensors"
    tensors = {}
    with safe_open(tensors_path, framework="pt", device="cpu") as f:
        for key in ("patches", "targets", "nn_idx"):
            tensors[key] = f.get_tensor(key)
    # compute the indices
    n_samples = tensors["patches"].size(0)
    n_train = int(n_samples * config.train_size)
    n_val = int(n_samples * config.val_size)
    # crop out the tensors
    train_p = tensors["patches"][:n_train]
    train_t = tensors["targets"][:n_train]
    train_p_idx = tensors["nn_idx"][:n_train]
    val_p = tensors["patches"][n_train:n_train+n_val]
    val_t = tensors["targets"][n_train:n_train+n_val]
    val_p_idx = tensors["nn_idx"][n_train:n_train+n_val]
    test_p = tensors["patches"][n_train+n_val:]
    test_t = tensors["targets"][n_train+n_val:]
    test_p_idx = tensors["nn_idx"][n_train+n_val:]
    # construct a dataset object
    train_ds = TemplateDataset(
        name=tensors_path.stem,
        patches=train_p,
        targets=train_t,
        train_patches=train_p,
        nn_idx=train_p_idx
    )
    val_ds = TemplateDataset(
        name=tensors_path.stem,
        patches=val_p,
        targets=val_t,
        train_patches=train_p,
        nn_idx=val_p_idx
    )
    test_ds = TemplateDataset(
        name=tensors_path.stem,
        patches=test_p,
        targets=test_t,
        train_patches=train_p,
        nn_idx=test_p_idx
    )
    # return the datasets
    return train_ds, val_ds, test_ds
