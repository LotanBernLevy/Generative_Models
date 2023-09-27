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
import torchvision


from experiments.Experiment import Experiment
from models.BaseModel import BaseModel
from data.datasets import MNISTDataModule


BUILD_DATA_ONLY_IF_NOT_EXIST = True
DATA_SIZE = 3000
VEC_LEN = 2
RANGE = (-1, 1)


class MNISTGenExperiment(Experiment):

    def on_validation_epoch_end(self):
        pass

    def reconstructed_examples(self, size:int=128, save_path:str=None):
        pass


def get_data(data_path):


    # create dataloaders

    data = MNISTDataModule(data_path, val_split=0.2)
    data.setup()

    return data










      