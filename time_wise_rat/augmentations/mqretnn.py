from time_wise_rat.augmentations import BaselineDataModule
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

import numpy as np


class MQRetNNDataModule(BaselineDataModule):

    def update_loaders(self) -> None:
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
            bs = batch.size(0)
            batch_embs = self.trainer.model.encode(batch.to(device)).to("cpu").view(bs, -1).numpy()
            embeddings.append(batch_embs)
        embeddings = np.concatenate(embeddings, axis=0)
        # get 10 closest matches by pearson's correlation
        n_train = len(self.train_ds)
        n_val = len(self.val_ds)
        train_embs = embeddings[:n_train]
        neigh = NearestNeighbors(
            n_neighbors=10,
            metric=lambda x, y: pearsonr(x, y)[0]
        )
        neigh.fit(train_embs)
        nn_idx = neigh.kneighbors(embeddings, return_distance=False)
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

    def val_dataloader(self):
        # only update nearest neighbors after pre-training
        if self.trainer.current_epoch == 0:
            self.update_loaders()
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

import torch
