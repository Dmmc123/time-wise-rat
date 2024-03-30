from time_wise_rat.data.preprocessors import TableToTensorPreprocessor
from hydra.core.config_store import ConfigStore
from time_wise_rat.configs import DataConfig


import hydra


cs = ConfigStore.instance()
cs.store(name="data", node=DataConfig)


@hydra.main(version_base=None, config_name="data")
def run_preprocessing(cfg: DataConfig) -> None:
    preprocessor = TableToTensorPreprocessor(config=cfg)
    preprocessor.run()


if __name__ == "__main__":
    run_preprocessing()
