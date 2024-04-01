from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
import numpy as np
import torch
import faiss


class TERA(pl.LightningDataModule):
    pass

#
# class TERA(BaselineRA):
#
#     def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         # get a dataloader of all time series in dataset
#         time_series = torch.concatenate(
#             tensors=(self.train_ds.samples, self.val_ds.samples, self.test_ds.samples),
#             dim=0
#         )
#         ts_ds = TensorDataset(time_series)
#         ts_dl = DataLoader(
#             dataset=ts_ds,
#             batch_size=self.cfg.train.batch_size,
#             num_workers=self.cfg.train.num_workers,
#             shuffle=True
#         )
#         # get embeddings from encoder of the model
#         embeddings = []
#         for batch in ts_dl:
#             bs = batch.size(0)
#             batch_embs = pl_module.encode(batch).to("cpu").view(bs, -1).numpy()
#             embeddings.append(batch_embs)
#         embeddings = np.stack(embeddings, axis=0)
#         # get nns for train part
#         n_train = len(self.train_ds)
#         n_val = len(self.val_ds)
#         train_embs = embeddings[:n_train]
#         index = faiss.IndexFlatIP(train_embs.shape[1])
#         index.add(train_embs)
#         _, nn_idx = index.search(embeddings, k=24)
#
#         # replace the content in  current datasets
#         train_embs = torch.tensor(train_embs, dtype=torch.float)
#         nn_idx = torch.tensor(nn_idx, dtype=torch.long)
#         self.train_ds.nn_embs = train_embs
#         self.train_ds.nn_idx = nn_idx[:n_train]
#         self.val_ds.nn_embs = train_embs
#         self.val_ds.nn_emb = nn_idx[n_train:n_train+n_val]
#         self.test_ds.nn_embs = train_embs
#         self.test_ds.nn_emb = nn_idx[n_train+n_val:]
#         # reload the dataloaders
#         train_dl, val_dl, test_dl = self.dm.get_dataloaders(
#             train_ds=self.train_ds,
#             val_ds=self.val_ds,
#             test_ds=self.test_ds
#         )
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#         self.test_dl = test_dl
#         # put new dataloaders in the trainer
#         trainer.train_dataloader = train_dl
#
#
# class TERADataModule(pl.LightningDataModule):
#     pass





