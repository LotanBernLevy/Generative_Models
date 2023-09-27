




### Adding new experiment loaders:
The experiments data should be provided as a lightning.pytorch.LightningDataModule, which provides getters to the train, val and test dataloaders. 
For example look for the RandomDataModule in dataset.py file.

