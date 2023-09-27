import torch
from typing import Any
from abc import abstractmethod


class BaseModel(torch.nn.Module):

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def sample(self, size):
        pass

    @abstractmethod
    def reconstruct(self,x):
        pass
