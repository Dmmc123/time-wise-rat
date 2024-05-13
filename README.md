# Transformer-Enabled Retrieval Augmentation

## Introduction

Well, well, well.

Who would have guessed tha [RAGs](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) would help Time Series models achieve a better prediction accuracy?

I guess, a lot of people.

## Usage

### Installation

```bash
poetry install
```

### Replication

```bash
python train.py -m
    model=patchtst,fedformer,autoformer,nstransformer,
    data=btc,sp_500,electrocity,flu,traffic,weather
    aug=baseline,mqretnn,ratsf,tera
```

## Results

### Experiments

![image](https://github.com/Dmmc123/time-wise-rat/assets/54360024/e43988dd-dcc9-44c6-8bbe-155619af761f)

### Summary

![image](https://github.com/Dmmc123/time-wise-rat/assets/54360024/49ac3d92-b812-496b-8775-e9d0264d0294)

