from datetime import datetime
import numpy as np
import torch
import lightning.pytorch as pl
from typing import Any, TYPE_CHECKING, Union
from abc import abstractmethod
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import utils
import torchvision
import plot_utils


from experiments.Experiment import Experiment
from models.BaseModel import BaseModel
from data.datasets import MNISTDataModule


BUILD_DATA_ONLY_IF_NOT_EXIST = True
DATA_SIZE = 3000
VEC_LEN = 2
RANGE = (-1, 1)


class MNISTGenExperiment(Experiment):

        def forward(self, inputs: torch.Tensor, labels: Union[int, torch.Tensor]) -> Any:
            return self.model(inputs, labels)

        def on_fit_start(self):
            super(MNISTGenExperiment, self).on_fit_start()
            self.epoch_num = 0

        def on_train_epoch_start(self):
            super(MNISTGenExperiment, self).on_train_epoch_start()

            self.epoch_num += 1


        def training_step(self, batch, batch_idx):
            
            
            inputs, label = batch
            outputs = self(inputs, label)

            loss_dict = self.model.loss(*outputs, batch_idx = batch_idx)
            self.log_dict({loss_name: loss.item() for loss_name, loss in loss_dict.items()}, on_step=False, on_epoch=True, sync_dist=True)
            return loss_dict['loss']


        def validation_step(self, batch, batch_idx):
            vinputs, label = batch
            voutputs = self(vinputs, label)

            vloss_dict = self.model.loss(*voutputs, batch_idx = batch_idx)
            self.log_dict({f"val_{loss_name}": loss.item() for loss_name, loss in vloss_dict.items()}, on_step=False, on_epoch=True, sync_dist=True)


            


        def on_validation_epoch_end(self):
            super(MNISTGenExperiment, self).on_validation_epoch_end()
            self.reconstructed_examples(size=5, save_path=self.logger.log_dir)


        def reconstructed_examples(self, size:int=5, save_path:str=None):

            if save_path is not None:
                save_dir = os.path.join(save_path, "reconstructed_images")
                Path(save_dir).mkdir(exist_ok=True, parents=True)
                
            imgs, labels = torch.tensor([]), torch.tensor([])

            for batch in iter(self.trainer.datamodule.test_dataloader()):
                imgs = torch.concatenate([imgs, batch[0]])
                labels = torch.cat([labels, batch[1]])
                if imgs.shape[0] >= size:
                    break

            test_samples_num = min(imgs.shape[0], size)

            save_path = os.path.join(save_dir, f"test_reconstructed_for_epoch_{self.epoch_num}.png")
            test_reconstructed = self.model.reconstruct(imgs[:test_samples_num], labels[:test_samples_num]).detach().numpy() 
            plot_utils.display_images(test_reconstructed, labels[:test_samples_num].detach().numpy() , save_path=save_path)


            save_path = os.path.join(save_dir, f"sampled_reconstructed_for_epoch_{self.epoch_num}.png")
            sampled_labels = np.arange(10)
            sampled_reconstructed = self.model.sample(10, torch.tensor(sampled_labels)).detach().numpy()
            plot_utils.display_images(sampled_reconstructed, sampled_labels, save_path=save_path)


            return test_reconstructed


def get_data(data_path):


    # create dataloaders

    data = MNISTDataModule(data_path, val_split=0.95)
    data.setup()

    return data










      