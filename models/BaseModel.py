import torch
from typing import Any
from abc import abstractmethod
from pytorch_lightning.core.mixins import HyperparametersMixin




class BaseModel(torch.nn.Module, HyperparametersMixin):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, *inputs: Any, **kwargs) -> dict:
        """Calculates the loss for the giving inputs

        Returns:
            dict: Dictionary with the losses values.
        """
        pass
    
    @abstractmethod
    def sample(self, size:int, *args: Any, **kwargs)->torch.tensor:
        """Sample a random items from P(x)

        Args:
            size (int): The number of items to sample
        Returns:
            torch.tensor: The sampled items.
        """
        pass

    @abstractmethod
    def reconstruct(self, *inputs: Any, **kwargs)->torch.tensor:
        """ Reconstruct the giving inputs
        Returns:
            torch.tensor: The reconstructed items.
        """
