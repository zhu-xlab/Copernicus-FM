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
import pdb

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class CoBenchAirQualityS5P(NonGeoDataset):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l3_airquality_s5p/airquality_s5p.zip"
    splits = ('train', 'val', 'test')
    split_fnames = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }
    # target stats for training set
    label_stats = {
        'no2': {'mean': 5.3167, 'std': 3.9948},
        'o3': {'mean': 4654.2632, 'std': 2589.4207},
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        modality = 'no2', # or 'o3'
        mode = 'annual', # or 'seasonal'
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self.modality = modality
        self.mode = mode

        if self.mode == 'annual':
            mode_dir = 's5p_annual'
        elif self.mode == 'seasonal':
            mode_dir = 's5p_seasonal'

        self.img_dir = os.path.join(root, modality, mode_dir)
        self.label_dir = os.path.join(root, modality, 'label_annual')
        
        self.split_csv = os.path.join(self.root, modality, self.split_fnames[split])
        with open(self.split_csv, 'r') as f:
            lines = f.readlines()
            self.pids = []
            for line in lines:
                self.pids.append(line.strip())

        self.label_mean = self.label_stats[modality]['mean']
        self.label_std = self.label_stats[modality]['std']

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (4*1)**2 # patchsize 4 pix, gsd 1km

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        label = self._load_target(index)
        if self.mode == 'annual':
            sample = {'image': images[0], 'groundtruth': label, 'meta': meta_infos[0]}
        elif self.mode == 'seasonal':
            sample = {'image': images, 'groundtruth': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        pid = self.pids[index]
        s5p_path = os.path.join(self.img_dir, pid)

        img_fnames = os.listdir(s5p_path)
        s5p_paths = []
        for img_fname in img_fnames:
            s5p_paths.append(os.path.join(s5p_path, img_fname))
        
        imgs = []
        meta_infos = []
        for img_path in s5p_paths:
            with rasterio.open(img_path) as src:
                img = src.read(1)
                img[np.isnan(img)] = 0
                img = cv2.resize(img, (56,56), interpolation=cv2.INTER_CUBIC)
                img = torch.from_numpy(img).float()
                img = img.unsqueeze(0)

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
                date_str = img_fname.split('_')[0][:10]
                date_obj = date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))
                delta = (date_obj - self.reference_date).days
                meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
                meta_info = torch.from_numpy(meta_info)

            imgs.append(img)
            meta_infos.append(meta_info)

        if self.mode == 'seasonal':
            # pad to 4 images if less than 4
            while len(imgs) < 4:
                imgs.append(img)
                meta_infos.append(meta_info)

        return imgs, meta_infos

    def _load_target(self, index):

        pid = self.pids[index]
        label_path = os.path.join(self.label_dir, pid+'.tif')

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label = cv2.resize(label, (56,56), interpolation=cv2.INTER_NEAREST) # 0-650
            # label contains -inf
            #pdb.set_trace()
            label[label<-1e10] = np.nan
            label[label>1e10] = np.nan

            #label = (label - self.label_mean) / self.label_std # normalize target

            label = torch.from_numpy(label.astype('float32'))

        return label



class RegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, band_stats, target_stats):
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

        if target_stats is not None:
            mean_target = torch.Tensor(target_stats['mean'])
            std_target = torch.Tensor(target_stats['std'])
            self.norm_target = K.augmentation.Normalize(mean=mean_target, std=std_target)


        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                #K.augmentation.RandomResizedCrop((56, 56), scale=(0.8, 1.0)),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "input"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "input"],
            )

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x,target = batch["image"], batch["groundtruth"]
        x = self.norm(x)
        if self.norm_target:
            target = self.norm_target(target)
            target = target.squeeze(0)
        x_out, target_out = self.transform(x, target.unsqueeze(0))
        return x_out.squeeze(0), target_out.squeeze(0), batch["meta"]


class CoBenchAirQualityS5PDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.modality = config.modality
        self.mode = config.mode
        self.band_stats = config.band_stats
        self.target_stats = config.target_stats

    def create_dataset(self):
        train_transform = RegDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats, target_stats=self.target_stats)
        eval_transform = RegDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats, target_stats=self.target_stats)

        dataset_train = CoBenchAirQualityS5P(
            root=self.root_dir, split="train", modality=self.modality, mode=self.mode, transforms=train_transform
        )
        dataset_val = CoBenchAirQualityS5P(
            root=self.root_dir, split="val", modality=self.modality, mode=self.mode, transforms=eval_transform
        )
        dataset_test = CoBenchAirQualityS5P(
            root=self.root_dir, split="test", modality=self.modality, mode=self.mode, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test