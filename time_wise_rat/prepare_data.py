from time_wise_rat.datasets import TSProcessor
from time_wise_rat.config import RatConfig
from pathlib import Path

import tqdm


def convert_csv_datasets_to_tensors(csv_dir: Path, cache_dir: Path, config: RatConfig, verbose: bool) -> None:
    # get all csv files in specified directory
    csv_files = csv_dir.glob("*.csv")
    # handle verbosity
    if verbose:
        csv_files = tqdm.tqdm(csv_files, desc="Reading CSV datasets")
    # run preprocessing jobs
    for csv_file in csv_files:
        TSProcessor.run(csv_path=csv_file, cache_dir=cache_dir, config=config)


if __name__ == "__main__":
    convert_csv_datasets_to_tensors(
        csv_dir=Path("data/raw"),
        cache_dir=Path("data/cache"),
        config=RatConfig(),
        verbose=True
    )
