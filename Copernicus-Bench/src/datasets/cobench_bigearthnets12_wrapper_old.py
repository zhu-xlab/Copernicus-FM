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
from torchgeo.datasets import BigEarthNet

from pyproj import Transformer
from datetime import date
import numpy as np
import pdb


class CoBenchBigEarthNetS12(BigEarthNet):
    splits_metadata = {
        "train": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet", # full data
            "url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/metadata-5%25.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
            "md5": "",  # unknown
        },
        "validation": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet", # full data
            "url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/metadata-5%25.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
            "md5": "",  # unknown
        },
        "test": {
            #"url": "https://zenodo.org/records/10891137/files/metadata.parquet", # full data
            "url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/metadata-5%25.parquet",
            "filename": "metadata-5%.parquet", #"metadata-10%.parquet",
            "md5": "",  # unknown
        },
    }
    metadata_locs = {
        "s1": {
            #"url": "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst",
            "url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/bigearthnetv2-s1-10%25.tar.zst",
            "directory": "BigEarthNet-S1-10%",
            "md5": "",  # unknown
        },
        "s2": {
            #"url": "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst",
            "url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/bigearthnetv2-s2-10%25.tar.zst",
            "directory": "BigEarthNet-S2-10%",
            "md5": "",  # unknown
        },
        "maps": {
            #"url": "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_bigearthnet_s1s2/reference_maps-10%25.tar.zst",
            "directory": "Reference_Maps-10%",
            "md5": "",  # unknown
        }, # not using maps
    }
    image_size = (120, 120)
    all_band_names_s1 = ('VV','VH')
    all_band_names_s2 = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        band_names: Sequence[str] = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:

        assert split in self.splits_metadata
        assert bands in ['s1', 's2']
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms


        self.band_names = band_names
        if self.bands == 's1':
            self.all_band_names = self.all_band_names_s1
        else:
            self.all_band_names = self.all_band_names_s2
        self.band_indices = [(self.all_band_names.index(b)+1) for b in band_names if b in self.all_band_names]

        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])}
        #self._verify()

        self.folders = self._load_folders()

        self.patch_area = (16*10/1000)**2
        self.reference_date = date(1970, 1, 1)

    def get_class2idx(self, label: str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def _load_folders(self) -> list[dict[str, str]]:
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        # dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))

        self.metadata = self.metadata[
            self.metadata['split'] == self.split
        ].reset_index(drop=True)

        def construct_folder_path(root, dir, patch_id, remove_last: int = 2):
            tile_id = "_".join(patch_id.split("_")[:-remove_last])
            return os.path.join(root, dir, tile_id, patch_id)

        folders = [
            {
                "s1": construct_folder_path(self.root, dir_s1, row["s1_name"], 3),
                "s2": construct_folder_path(self.root, dir_s2, row["patch_id"], 2),
                # "maps": construct_folder_path(self.root, dir_maps, row["patch_id"], 2),
            }
            for _, row in self.metadata.iterrows()
        ]

        return folders

    def _load_target(self, index: int) -> Tensor:
        image_labels = self.metadata.iloc[index]["labels"]

        # labels -> indices
        indices = [
            self.get_class2idx(label, level=self.num_classes) for label in image_labels
        ]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target

    def _load_paths(self, index: int) -> list[str]:
        if self.bands == 's1':
            folder = self.folders[index]['s1']
            paths = glob.glob(os.path.join(folder, '*_allbands.tif'))
            paths = sorted(paths)
        else:
            folder = self.folders[index]['s2']
            paths = glob.glob(os.path.join(folder, '*_allbands.tif'))
            paths = sorted(paths)

        return paths

    def _load_image(self, index: int) -> Tensor:
        paths = self._load_paths(index)
        #images = []
        for path in paths:
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

                if self.bands == 's1':
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

    def __init__(self, split, size, bands, band_stats):
        super().__init__()

        self.bands = bands

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
        if self.bands == "rgb":
            sample["image"] = sample["image"][1:4, ...].flip(dims=(0,))
            # get in rgb order and then normalization can be applied
        x_out = self.transform(sample["image"]).squeeze(0)
        return x_out, sample["label"], sample["meta"]


class ClsDataAugmentationSoftCon(torch.nn.Module):

    def __init__(self, split, size, bands, band_stats):
        super().__init__()

        self.bands = bands

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
        if self.bands == 's1':
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
        elif self.bands == 's2':
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
        self.bands = config.modality
        self.band_names = config.band_names
        self.num_classes = config.num_classes
        self.band_stats = config.band_stats
        self.norm_form = config.norm_form if 'norm_form' in config else None

        if self.bands == "rgb":
            # start with rgb and extract later
            self.input_bands = "s2"
        else:
            self.input_bands = self.bands

    def create_dataset(self):

        if self.norm_form == 'softcon':
            train_transform = ClsDataAugmentationSoftCon(
                split="train", size=self.img_size, bands=self.bands, band_stats=self.band_stats
            )
            eval_transform = ClsDataAugmentationSoftCon(
                split="test", size=self.img_size, bands=self.bands, band_stats=self.band_stats
            )
        else:
            train_transform = ClsDataAugmentation(
                split="train", size=self.img_size, bands=self.bands, band_stats=self.band_stats
            )
            eval_transform = ClsDataAugmentation(
                split="test", size=self.img_size, bands=self.bands, band_stats=self.band_stats
            )

        dataset_train = CoBenchBigEarthNetS12(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="train",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=train_transform,
        )

        dataset_val = CoBenchBigEarthNetS12(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="validation",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=eval_transform,
        )
        dataset_test = CoBenchBigEarthNetS12(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="test",
            bands=self.input_bands,
            band_names=self.band_names,
            transforms=eval_transform,
        )

        return dataset_train, dataset_val, dataset_test
