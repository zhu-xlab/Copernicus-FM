import kornia.augmentation as K
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

class CoBenchLC100ClsS3(NonGeoDataset):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_lc100_s3/lc100_s3_v0.zip"
    splits = ('train', 'val', 'test')
    label_filenames = {
        'train': 'lc100_multilabel-train.csv',
        'val': 'lc100_multilabel-val.csv',
        'test': 'lc100_multilabel-test.csv',
    }
    static_filenames = {
        'train': 'static_fnames-train.csv',
        'val': 'static_fnames-val.csv',
        'test': 'static_fnames-test.csv',
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

    LC100_CLSID = {
        0: 0, # unknown
        20: 1,
        30: 2,
        40: 3,
        50: 4,
        60: 5,
        70: 6,
        80: 7,
        90: 8,
        100: 9,
        111: 10,
        112: 11,
        113: 12,
        114: 13,
        115: 14,
        116: 15,
        121: 16,
        122: 17,
        123: 18,
        124: 19,
        125: 20,
        126: 21,
        200: 22, # ocean
    }


    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        mode = 'static',
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
        self.lc100_cls = os.path.join(self.root, self.label_filenames[split])

        self.pids = []
        self.labels = []
        with open(self.lc100_cls, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.pids.append(line.strip().split(',')[0])
                self.labels.append(list(map(int, line.strip().split(',')[1:])))
        
        if self.mode == 'static':
            self.static_csv = os.path.join(self.root, self.static_filenames[split])
            with open(self.static_csv, 'r') as f:
                lines = f.readlines()
                self.static_img = {}
                for line in lines:
                    pid = line.strip().split(',')[0]
                    img_fname = line.strip().split(',')[1]
                    self.static_img[pid] = img_fname


        self.reference_date = date(1970, 1, 1)
        self.patch_area = (8*300/1000)**2 # patchsize 8 pix, gsd 300m

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        #meta_info = np.array([coord[0], coord[1], np.nan, self.patch_area]).astype(np.float32)
        label = self._load_target(index)
        if self.mode == 'static':
            sample = {'image': images[0], 'label': label, 'meta': meta_infos[0]}
        elif self.mode == 'series':
            sample = {'image': images, 'label': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        pid = self.pids[index]
        s3_path = os.path.join(self.img_dir, pid)
        if self.mode == 'static':
            img_fname = self.static_img[pid]
            s3_paths = [os.path.join(s3_path, img_fname)]
        else:
            img_fnames = os.listdir(s3_path)
            s3_paths = []
            for img_fname in img_fnames:
                s3_paths.append(os.path.join(s3_path, img_fname))
        
        imgs = []
        img_paths = []
        meta_infos = []
        for img_path in s3_paths:
            with rasterio.open(img_path) as src:
                img = src.read()
                img[np.isnan(img)] = 0
                chs = []
                for b in range(21):
                    ch = img[b]*self.all_band_scale[b]
                    ch = cv2.resize(ch, (96,96), interpolation=cv2.INTER_CUBIC)
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
                img_fname = os.path.basename(img_path)
                date_str = img_fname.split('_')[1][:8]
                date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                delta = (date_obj - self.reference_date).days
                meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
                meta_info = torch.from_numpy(meta_info)

            imgs.append(img)
            img_paths.append(img_path)
            meta_infos.append(meta_info)

        if self.mode == 'series':
            # pad to 4 images if less than 4
            while len(imgs) < 4:
                imgs.append(img)
                img_paths.append(img_path)
                meta_infos.append(meta_info)

        return imgs, meta_infos

    def _load_target(self, index):

        label = self.labels[index]
        labels = torch.zeros(23)
        # turn into one-hot
        for l in label:
            cls_id = self.LC100_CLSID[l]
            labels[cls_id] = 1

        return labels


class ClsDataAugmentation(torch.nn.Module):

    def __init__(self, split, size):
        super().__init__()

        mean = torch.Tensor([0.0])
        std = torch.Tensor([1.0])

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
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"], batch["meta"]


class CoBenchLC100ClsS3Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.mode = config.mode

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size)

        dataset_train = CoBenchLC100ClsS3(
            root=self.root_dir, split="train", bands=self.bands, mode=self.mode, transforms=train_transform
        )
        dataset_val = CoBenchLC100ClsS3(
            root=self.root_dir, split="val", bands=self.bands, mode=self.mode, transforms=eval_transform
        )
        dataset_test = CoBenchLC100ClsS3(
            root=self.root_dir, split="test", bands=self.bands, mode=self.mode, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
