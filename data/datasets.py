from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
import numpy as np
import torch
from typing import Optional
import torchvision

class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class RandomInRect(MyDataset):

    def __init__(self, random_data_path, split="train", train_ratio=0.75):
        npdata = np.loadtxt(random_data_path, delimiter=",", dtype=float)
        self.data = torch.Tensor(npdata[:int(len(npdata)*train_ratio)] if split == "train" else npdata[int(len(npdata)*train_ratio):])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],0.0


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False,
        **kwargs,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.val_data = RandomInRect(self.data_dir, split="val")
        self.train_data = RandomInRect(self.data_dir, split="train")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=128, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 0, pin_memory: bool = False, val_split:int = None,
        **kwargs,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = torchvision.datasets.MNIST(self.data_dir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

        self.test_data = torchvision.datasets.MNIST(self.data_dir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

        if self.val_split is not None:
            val_length = min(int((len(self.train_data) + len(self.test_data)) * self.val_split), len(self.test_data))
            test_length = max(0, len(self.test_data) - val_length)
            print(val_length, test_length, len(self.test_data))

            self.test_data = torch.utils.data.random_split(self.test_data, [val_length, test_length])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        val_data = self.test_data
        if isinstance(self.test_data, tuple):
            val_data = self.test_data[0]
        return DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        test_data = self.test_data
        if isinstance(self.test_data, tuple):
            test_data = self.test_data[1]
        return DataLoader(test_data, batch_size=128, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)