from time_wise_rat.augmentations import BaselineDataModule
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import torch
import faiss


class TERADataModule(BaselineDataModule):

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
            batch_embs = self.trainer.model.encode(batch.to(device)).to("cpu")
            if len(batch_embs.size()) == 3:
                batch_embs = batch_embs.mean(dim=1)
            embeddings.append(batch_embs.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        # get nns for train part
        n_train = len(self.train_ds)
        n_val = len(self.val_ds)
        train_embs = embeddings[:n_train]
        index = faiss.IndexFlatIP(train_embs.shape[1])
        index.add(train_embs)
        _, nn_idx = index.search(embeddings, k=24)
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
        self.update_loaders()
        return self.val_dl

    def test_dataloader(self):
        self.update_loaders()
        return self.test_dl
