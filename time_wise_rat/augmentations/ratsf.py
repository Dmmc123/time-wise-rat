from time_wise_rat.augmentations import BaselineDataModule
from torch.utils.data import DataLoader, TensorDataset
from numba import jit

import pynndescent
import numpy as np
import torch
import faiss


@jit(nopython=True)
def dtw_distance_numba(s: np.ndarray, t: np.ndarray) -> float:
    n, m = len(s), len(t)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],    # insertion
                cost_matrix[i, j - 1],    # deletion
                cost_matrix[i - 1, j - 1] # match
            )

    return cost_matrix[n, m]


class RATSFDataModule(BaselineDataModule):

    def update_loaders_dtw(self) -> None:
        # get a dataloader of all time series in dataset
        time_series = torch.concatenate(
            tensors=(self.train_ds.samples, self.val_ds.samples, self.test_ds.samples),
            dim=0
        )
        if len(time_series.size()) == 3:
            ts_block_1 = time_series[:, :, 0]    # B x W
            ts_block_2 = time_series[:, -1, 1:]  # B x (D - 1)
            time_series = torch.cat([ts_block_1, ts_block_2], dim=1)
        time_series = time_series.numpy()
        # get 3 closest matches by DTW
        n_train = len(self.train_ds)
        n_val = len(self.val_ds)
        train_ts = time_series[:n_train]
        index = pynndescent.NNDescent(train_ts, metric=dtw_distance_numba)
        index.prepare()
        nn_idx, _ = index.query(time_series, k=3)
        # replace the content in  current datasets
        train_samples = self.train_ds.samples
        nn_idx = torch.tensor(nn_idx, dtype=torch.long)
        self.train_ds.nn_emb = train_samples
        self.train_ds.nn_idx = nn_idx[:n_train]
        self.val_ds.nn_emb = train_samples
        self.val_ds.nn_idx = nn_idx[n_train:n_train+n_val]
        self.test_ds.nn_emb = train_samples
        self.test_ds.nn_idx = nn_idx[n_train+n_val:]
        # reload the dataloaders
        train_dl, val_dl, test_dl = self.data_manager.get_dataloaders(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            test_ds=self.test_ds
        )
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    def update_loaders_l2(self) -> None:
        # get a dataloader of all time series in dataset
        time_series = torch.concatenate(
            tensors=(self.train_ds.samples, self.val_ds.samples, self.test_ds.samples),
            dim=0
        )
        ts_ds = TensorDataset(time_series)
        ts_dl = DataLoader(
            dataset=ts_ds,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True
        )
        # get embeddings from encoder of the model
        embeddings = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for (batch,) in ts_dl:
            batch_embs = self.trainer.model.encode(batch.to(device)).to("cpu")
            if len(batch_embs.size()) == 3:
                batch_embs = batch_embs.mean(dim=1)
            embeddings.append(batch_embs.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        # get nns for train part
        n_train = len(self.train_ds)
        n_val = len(self.val_ds)
        train_embs = embeddings[:n_train]
        index = faiss.IndexFlatL2(train_embs.shape[1])  # euclidean
        index.add(train_embs)
        _, nn_idx = index.search(embeddings, k=3)
        # replace the content in  current datasets
        train_samples = self.train_ds.samples
        nn_idx = torch.tensor(nn_idx, dtype=torch.long)
        self.train_ds.nn_emb = train_samples
        self.train_ds.nn_idx = nn_idx[:n_train]
        self.val_ds.nn_emb = train_samples
        self.val_ds.nn_idx = nn_idx[n_train:n_train+n_val]
        self.test_ds.nn_emb = train_samples
        self.test_ds.nn_idx = nn_idx[n_train+n_val:]
        # reload the dataloaders
        train_dl, val_dl, test_dl = self.data_manager.get_dataloaders(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            test_ds=self.test_ds
        )
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    def train_dataloader(self):
        if self.trainer.current_epoch == 0:
            self.update_loaders_dtw()
        elif self.trainer.current_epoch == 1:
            self.update_loaders_l2()
        return self.train_dl
