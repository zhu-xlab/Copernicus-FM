import kornia as K
import torch
from torchgeo.datasets.geo import NonGeoDataset
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
import cv2
from pyproj import Transformer
from datetime import date
from typing import TypeAlias, ClassVar
import pathlib
from shapely import wkt
import pandas as pd

import logging
import pdb

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class CoBenchCloudS2(NonGeoDataset):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l1_cloud_s2/cloud_s2.zip"
    #base_dir = ""
    all_band_names = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')

    split_filenames = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }

    Cls_index_multi = {
        'clear': 0,
        'thick cloud': 1,
        'thin cloud': 2,
        'cloud shadow': 3,
    }



    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self.bands = bands
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        #self.root = os.path.join(self.root, self.base_dir)

        self.img_dir = os.path.join(self.root, 's2_toa')
        self.label_dir = os.path.join(self.root, 'cloud')

        self.split_csv = os.path.join(self.root, self.split_filenames[split])
        self.fnames = []
        with open(self.split_csv, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = line.strip()
                self.fnames.append(fname)
        
        self.reference_date = date(1970, 1, 1)
        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        label = self._load_target(index)

        sample = {'image': images, 'mask': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        fname = self.fnames[index]
        img_path = os.path.join(self.img_dir, fname+'.tif')
        
        with rasterio.open(img_path) as src:
            img = src.read(self.band_indices).astype('float32')
            img = torch.from_numpy(img)

            # get lon, lat
            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                # convert to lon, lat
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy
            # get time
            img_fname = os.path.basename(img_path)
            date_str = img_fname.split('__')[1][:8]
            date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            delta = (date_obj - self.reference_date).days
            meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
            meta_info = torch.from_numpy(meta_info)

        return img, meta_info

    def _load_target(self, index):

        fname = self.fnames[index]
        label_path = os.path.join(self.label_dir, fname+'.tif')

        with rasterio.open(label_path) as src:
            label = src.read(1)
            labels = torch.from_numpy(label).long()

        return labels



class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, band_stats):
        super().__init__()

        if band_stats is not None:
            mean = band_stats['mean']
            std = band_stats['std']
        else:
            mean = [0.0]
            std = [1.0]

        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        self.norm = K.augmentation.Normalize(mean=mean, std=std)

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x,mask = batch["image"], batch["mask"]
        x = self.norm(x)
        x_out, mask_out = self.transform(x, mask)
        return x_out.squeeze(0), mask_out.squeeze(0).squeeze(0), batch["meta"]


class SegDataAugmentationSoftCon(torch.nn.Module):

    def __init__(self, split, size, band_stats):
        super().__init__()

        if band_stats is not None:
            self.mean = band_stats['mean']
            self.std = band_stats['std']
        else:
            self.mean = [0.0]
            self.std = [1.0]

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                #K.augmentation.RandomResizedCrop(size=size, scale=(0.8,1.0)),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        sample_img,mask = sample["image"], sample["mask"]

        img_bands = []
        for b in range(13):
            img = sample_img[b,:,:].clone()
            ## normalize
            img = self.normalize(img,self.mean[b],self.std[b])         
            img_bands.append(img)
        sample_img = torch.stack(img_bands,dim=0)

        x_out, mask_out = self.transform(sample_img, mask)
        return x_out.squeeze(0), mask_out.squeeze(0).squeeze(0), sample["meta"]

    @torch.no_grad()
    def normalize(self, img, mean, std):
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        return img


class CoBenchCloudS2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.band_stats = config.band_stats
        self.norm_form = config.norm_form if 'norm_form' in config else None


    def create_dataset(self):
        if self.norm_form == 'softcon':
            train_transform = SegDataAugmentationSoftCon(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = SegDataAugmentationSoftCon(split="test", size=self.img_size, band_stats=self.band_stats)
        else:
            train_transform = SegDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = SegDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = CoBenchCloudS2(
            root=self.root_dir, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = CoBenchCloudS2(
            root=self.root_dir, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = CoBenchCloudS2(
            root=self.root_dir, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test