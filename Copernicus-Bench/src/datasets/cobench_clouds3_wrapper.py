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

import logging

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class CoBenchCloudS3(NonGeoDataset):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l1_cloud_s3/cloud_s3.zip"
    splits = ('train', 'val', 'test')

    split_filenames = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }
    all_band_names = (
        'Oa01_radiance', 'Oa02_radiance', 'Oa03_radiance', 'Oa04_radiance', 'Oa05_radiance', 'Oa06_radiance', 'Oa07_radiance',
        'Oa08_radiance', 'Oa09_radiance', 'Oa10_radiance', 'Oa11_radiance', 'Oa12_radiance', 'Oa13_radiance', 'Oa14_radiance',
        'Oa15_radiance', 'Oa16_radiance', 'Oa17_radiance', 'Oa18_radiance', 'Oa19_radiance', 'Oa20_radiance', 'Oa21_radiance',
    )
    all_band_scale = (
        0.0139465,0.0133873,0.0121481,0.0115198,0.0100953,0.0123538,0.00879161,
        0.00876539,0.0095103,0.00773378,0.00675523,0.0071996,0.00749684,0.0086512,
        0.00526779,0.00530267,0.00493004,0.00549962,0.00502847,0.00326378,0.00324118)
    rgb_bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')

    Cls_index_binary = {
        'invalid': 0, # --> 255 should be ignored during training
        'clear': 1, # --> 0
        'cloud': 2, # --> 1
    }

    Cls_index_multi = {
        'invalid': 0, # --> 255 should be ignored during training
        'clear': 1, # --> 0
        'cloud-sure': 2, # --> 1
        'cloud-ambiguous': 3, # --> 2
        'cloud shadow': 4, # --> 3
        'snow and ice': 5, # --> 4
    }



    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        mode = 'multi',
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

        self.mode = mode
        self.img_dir = os.path.join(self.root, 's3_olci')
        self.label_dir = os.path.join(self.root, 'cloud_'+mode)
        
        self.split_csv = os.path.join(self.root, self.split_filenames[split])
        self.fnames = []
        with open(self.split_csv, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = line.strip()
                self.fnames.append(fname)

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (8*300/1000)**2 # patchsize 8 pix, gsd 300m

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        #meta_info = np.array([coord[0], coord[1], np.nan, self.patch_area]).astype(np.float32)
        label = self._load_target(index)
        sample = {'image': images, 'mask': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        fname = self.fnames[index]
        s3_path = os.path.join(self.img_dir, fname)
        
        with rasterio.open(s3_path) as src:
            img = src.read(self.band_indices)
            img[np.isnan(img)] = 0
            chs = []
            for b in range(21):
                ch = img[b]*self.all_band_scale[b]
                #ch = cv2.resize(ch, (256,256), interpolation=cv2.INTER_CUBIC)
                chs.append(ch)
            img = np.stack(chs)
            img = torch.from_numpy(img).float()

            # get lon, lat
            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                # convert to lon, lat
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy
            # get time
            img_fname = os.path.basename(s3_path)
            date_str = img_fname.split('____')[1][:8]
            date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            delta = (date_obj - self.reference_date).days
            meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
            meta_info = torch.from_numpy(meta_info)

        return img, meta_info

    def _load_target(self, index):

        fname = self.fnames[index]
        label_path = os.path.join(self.label_dir, fname)

        with rasterio.open(label_path) as src:
            label = src.read(1)
            #label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST) # 0-650
            label[label==0] = 256
            label = label - 1
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


class CoBenchCloudS3Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.mode = config.mode
        self.band_stats = config.band_stats

    def create_dataset(self):
        train_transform = SegDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
        eval_transform = SegDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = CoBenchCloudS3(
            root=self.root_dir, split="train", bands=self.bands, mode=self.mode, transforms=train_transform
        )
        dataset_val = CoBenchCloudS3(
            root=self.root_dir, split="val", bands=self.bands, mode=self.mode, transforms=eval_transform
        )
        dataset_test = CoBenchCloudS3(
            root=self.root_dir, split="test", bands=self.bands, mode=self.mode, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test