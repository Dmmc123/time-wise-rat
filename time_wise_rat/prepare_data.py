from time_wise_rat.datasets import (
    TSProcessor,
    RATSProcessor,
    split_tensors_into_datasets,
    split_tensors_into_ra_datasets
)
from time_wise_rat.config import RatConfig

from pathlib import Path

import tqdm


def convert_csv_datasets_to_tensors(
        csv_dir: Path,
        cache_dir: Path,
        config: RatConfig,
        verbose: bool
) -> None:
    # get all csv files in specified directory
    csv_files = csv_dir.glob("*.csv")
    # handle verbosity
    if verbose:
        csv_files = tqdm.tqdm(csv_files, desc="Reading CSV datasets")
    # run preprocessing jobs
    for csv_file in csv_files:
        TSProcessor.run(csv_path=csv_file, cache_dir=cache_dir, config=config)


def enrich_tensors_with_neighbors(
        cache_dir: Path,
        weights_dir: Path,
        config: RatConfig,
        verbose: bool
) -> None:
    # get all tensors
    tensor_files = cache_dir.glob("*.safetensors")
    # handle verbosity
    if verbose:
        tensor_files = tqdm.tqdm(tensor_files, desc="Processing tensor datasets")
    # run preprocessing jobs
    for tensor_file in tensor_files:
        ckpt_files = list((weights_dir / "Baseline" / tensor_file.stem).glob("*.ckpt"))
        if len(ckpt_files) == 0:
            # print(f"Skipping dataset {tensor_file.stem}, no model")
            continue
        best_ckpt = min(ckpt_files, key=lambda p: float(p.stem.split("=")[-1]))
        RATSProcessor.run(
            cache_dir=cache_dir,
            name=tensor_file.stem,
            config=config,
            baseline_ckpt_path=best_ckpt
        )


if __name__ == "__main__":
    pass
    # enrich_tensors_with_neighbors(
    #     cache_dir=Path("data/cache"),
    #     weights_dir=Path("weights"),
    #     config=RatConfig(),
    #     verbose=True
    # )
    # for t in train_ds[-1]:
    #     print(t)
    # convert_csv_datasets_to_tensors(
    #     csv_dir=Path("data/raw"),
    #     cache_dir=Path("data/cache"),
    #     config=RatConfig(),
    #     verbose=True
    # )
    # train_ds, val_ds, test_ds = split_tensors_into_ra_datasets(
    #     cache_dir=Path("data/cache"),
    #     name="btc",
    #     config=RatConfig()
    # )
    # print(test_ds.nn_idx.size(), test_ds.train_patches.size())
    # for t in val_ds[-1]:
    #     print(t)