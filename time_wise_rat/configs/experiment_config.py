from dataclasses import dataclass, field

from time_wise_rat.configs import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    AugmentationConfig
)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)
