from dataclasses import dataclass, field

from time_wise_rat.configs import (
    DataConfig,
    ModelConfig,
    TrainConfig
)


@dataclass
class ExperimentConfig:
    data_cfg: DataConfig = field(default_factory=DataConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
