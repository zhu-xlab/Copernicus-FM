from lightning import LightningDataModule
import torch
from factory import create_dataset


class BenchmarkDataModule(LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers, pin_memory):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.dataset_train, self.dataset_val, self.dataset_test = create_dataset(
            self.dataset_config
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
