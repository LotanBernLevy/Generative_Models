from datetime import datetime
import numpy as np
import torch
import lightning.pytorch as pl
from typing import Any, TYPE_CHECKING
from abc import abstractmethod
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import utils


from experiments.Experiment import Experiment
from models.BaseModel import BaseModel
from data.datasets import RandomDataModule


BUILD_DATA_ONLY_IF_NOT_EXIST = True
DATA_SIZE = 3000
VEC_LEN = 2
RANGE = (-1, 1)


class RangeGenExperiment(Experiment):

    def on_validation_epoch_end(self):
        self.reconstructed_examples(size=128, save_path=self.logger.log_dir)

    def reconstructed_examples(self, size:int=128, save_path:str=None):
        
        test_input = torch.concatenate([batch[0] for batch in list(iter(self.trainer.datamodule.test_dataloader()))])[:size]
        test_reconstructed = self.model.reconstruct(test_input).detach().numpy()

        sampled_reconstructed = self.model.sample(size).detach().numpy()

        if save_path is not None:
            save_dir = os.path.join(save_path, "reconstructed_images")
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            save_path = os.path.join(save_dir, f"reconstructed_for_epoch_{self.current_epoch}.png")

            plt.figure()
            plt.title(f"epoch {self.current_epoch}")
            ax = plt.subplot(111)
            ax.scatter(test_reconstructed[:,0], test_reconstructed[:,1], c="b", label="test")
            ax.scatter(sampled_reconstructed[:,0], sampled_reconstructed[:,1], c="r", label="sampled")
            ax.vlines([-1,1], -1,1)
            ax.hlines([-1,1], -1,1)
            ax.legend()
            plt.tight_layout()

            plt.savefig(save_path)
            plt.close()
            
        return test_reconstructed, sampled_reconstructed


def get_data(data_path):
    # Builds random data and saves it
    if not BUILD_DATA_ONLY_IF_NOT_EXIST or not os.path.exists(data_path):

        utils.dprint("Build random data")

        if not os.path.exists(os.path.dirname(data_path)):
            Path(os.path.dirname(data_path)).mkdir(exist_ok=True, parents=True)
        data_arr = utils.get_unique_vecs(lambda size: np.random.uniform(*RANGE,size), VEC_LEN, DATA_SIZE)
        np.savetxt(data_path, data_arr, delimiter=',')

    # create dataloaders

    data = RandomDataModule(data_path)
    data.setup()

    return data










      