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


from models.BaseModel import BaseModel

class Experiment(pl.LightningModule):

    def __init__(self, model:BaseModel, learning_rate:float=0.001, weight_decay:float=0):
        super(Experiment, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])




    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> Any:
        return self.model(inputs)


    def training_step(self, batch, batch_idx):
        inputs, label = batch
        outputs = self(inputs)

        loss_dict = self.model.loss(*outputs, batch_idx = batch_idx)
        self.log_dict({loss_name: loss.item() for loss_name, loss in loss_dict.items()}, sync_dist=True)
        return loss_dict['loss']


    def validation_step(self, batch, batch_idx):

        
        vinputs, label = batch
        voutputs = self(vinputs)

        vloss_dict = self.model.loss(*voutputs, batch_idx = batch_idx)
        self.log_dict({f"val_{loss_name}": loss.item() for loss_name, loss in vloss_dict.items()}, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr= self.hparams.learning_rate,
                               weight_decay= self.hparams.weight_decay)

        return {"optimizer": optimizer}

    









      