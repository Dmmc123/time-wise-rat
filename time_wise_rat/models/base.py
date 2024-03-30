from abc import ABC, abstractmethod
from time_wise_rat.configs import (
    ModelConfig,
    DataConfig
)
from typing import Optional
from torch import Tensor


class BaselineModel(ABC):

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, x_emb: Tensor, x_cnt: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError()
