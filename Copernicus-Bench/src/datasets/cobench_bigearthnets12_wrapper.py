import glob
import os
from typing import Callable, Optional
from collections.abc import Sequence

import kornia.augmentation as K
import pandas as pd
import rasterio
import torch
from torch import Generator, Tensor
from torch.utils.data import random_split
#from torchgeo.datasets import BigEarthNet
from torchgeo.datasets.geo import NonGeoDataset

from pyproj import Transformer
from datetime import date
import numpy as np
import pdb
import ast


class CoBenchBigEarthNetS12(NonGeoDataset):
    url = 'https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/bigearthnetv2.zip'
    splits = ('train', 'val', 'test')
    label_filenames = {
        'train': 'multilabel-train.csv',
        'val': 'multilabel-val.csv',
        'test': 'multilabel-test.csv',
    }
    image_size = (120, 120)
    all_band_names_s1 = ('VV','VH')
    all_band_names_s2 = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        modality: str = "s2",
        bands: Sequence[str] = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:

        assert split in self.splits
        assert modality in ['s1', 's2']
        self.root = root
        self.split = split
        self.modality = modality
        self.transforms = transforms


        self.bands = bands
        if self.modality == 's1':
            self.all_band_names = self.all_band_names_s1
        else:
            self.all_band_names = self.all_band_names_s2
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        self.img_paths = []
        self.labels = []

        df = pd.read_csv(os.path.join(self.root,self.label_filenames[self.split]))
        s1_paths = df["s1_path"].tolist()
        s2_paths = df["s2_path"].tolist()
        if self.modality == 's1':
            self.img_paths = [os.path.join(self.root, 'BigEarthNet-S1-5%', path) for path in s1_paths]
        elif self.modality == 's2':
            self.img_paths = [os.path.join(self.root, 'BigEarthNet-S2-5%', path) for path in s2_paths]
        self.class_names = df.columns[2:]  # Exclude 's1_path' and 's2_path'
        self.binary_labels = df.iloc[:, 2:].values  # Convert label data to NumPy array


        self.patch_area = (16*10/1000)**2
        self.reference_date = date(1970, 1, 1)

    def __len__(self):
        return len(self.img_paths)

    def _load_target(self, index: int) -> Tensor:
        image_labels = self.binary_labels[index]
        image_target = torch.from_numpy(image_labels).long()

        return image_target

    def _load_image(self, index: int) -> Tensor:
        path = self.img_paths[index]
        # Bands are of different spatial resolutions
        # Resample to (120, 120)
        with rasterio.open(path) as src:
            array = src.read(
                self.band_indices,
            ).astype('float32')

            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy

            if self.modality == 's1':
                date_str = path.split('/')[-1].split('_')[4]
            else:
                date_str = path.split('/')[-1].split('_')[2]
            date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            delta = (date_obj - self.reference_date).days

        tensor = torch.from_numpy(array).float()
        return tensor, (lon,lat), delta

    def __getitem__(self, index: int) -> dict[str, Tensor]:

        image, coord, delta = self._load_image(index)
        meta_info = np.array([coord[0], coord[1], delta, self.patch_area]).astype(np.float32)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {'image': image, 'label': label, 'meta':meta_info}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample



class ClsDataAugmentation(torch.nn.Module):

    def __init__(self, split, size, modality, band_stats):
        super().__init__()

        self.modality = modality

        if band_stats is not None:
            mean = band_stats['mean']
            std = band_stats['std']
        else:
            mean = [0.0]
            std = [1.0]

        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple."""
        x_out = self.transform(sample["image"]).squeeze(0)
        return x_out, sample["label"], sample["meta"]


class ClsDataAugmentationSoftCon(torch.nn.Module):

    def __init__(self, split, size, modality, band_stats):
        super().__init__()

        self.modality = modality

        if band_stats is not None:
            self.mean = band_stats['mean']
            self.std = band_stats['std']
        else:
            self.mean = [0.0]
            self.std = [1.0]

        # mean = torch.Tensor(mean)
        # std = torch.Tensor(std)

        if split == "train":
            self.transform = torch.nn.Sequential(
                #K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                #K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple."""
        if self.modality == 's1':
            sample_img = sample["image"]
            ### normalize s1
            self.max_q = torch.quantile(sample_img.reshape(2,-1),0.99,dim=1)      
            self.min_q = torch.quantile(sample_img.reshape(2,-1),0.01,dim=1)
            img_bands = []
            for b in range(2):
                img = sample_img[b,:,:].clone()
                ## outlier
                max_q = self.max_q[b]
                min_q = self.min_q[b]            
                img = torch.clamp(img, min_q, max_q)
                ## normalize
                img = self.normalize(img,self.mean[b],self.std[b])         
                img_bands.append(img)
            sample_img = torch.stack(img_bands,dim=0) # VV,VH (w,h,c)
        elif self.modality == 's2':
            sample_img = sample["image"]
            img_bands = []
            for b in range(12):
                img = sample_img[b,:,:].clone()
                ## normalize
                img = self.normalize(img,self.mean[b],self.std[b])         
                img_bands.append(img)
                if b==9:
                    # pad zero to B10
                    img_bands.append(torch.zeros_like(img))
            sample_img = torch.stack(img_bands,dim=0)
            
        x_out = self.transform(sample_img).squeeze(0)
        return x_out, sample["label"], sample["meta"]

    @torch.no_grad()
    def normalize(self, img, mean, std):
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        return img


class CoBenchBigEarthNetS12Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.modality = config.modality
        self.bands = config.band_names
        self.band_stats = config.band_stats
        self.norm_form = config.norm_form if 'norm_form' in config else None

    def create_dataset(self):

        if self.norm_form == 'softcon':
            train_transform = ClsDataAugmentationSoftCon(
                split="train", size=self.img_size, modality=self.modality, band_stats=self.band_stats
            )
            eval_transform = ClsDataAugmentationSoftCon(
                split="test", size=self.img_size, modality=self.modality, band_stats=self.band_stats
            )
        else:
            train_transform = ClsDataAugmentation(
                split="train", size=self.img_size, modality=self.modality, band_stats=self.band_stats
            )
            eval_transform = ClsDataAugmentation(
                split="test", size=self.img_size, modality=self.modality, band_stats=self.band_stats
            )

        dataset_train = CoBenchBigEarthNetS12(
            root=self.root_dir,
            split="train",
            modality=self.modality,
            bands=self.bands,
            transforms=train_transform,
        )

        dataset_val = CoBenchBigEarthNetS12(
            root=self.root_dir,
            split="val",
            modality=self.modality,
            bands=self.bands,
            transforms=eval_transform,
        )
        dataset_test = CoBenchBigEarthNetS12(
            root=self.root_dir,
            split="test",
            modality=self.modality,
            bands=self.bands,
            transforms=eval_transform,
        )

        return dataset_train, dataset_val, dataset_test
