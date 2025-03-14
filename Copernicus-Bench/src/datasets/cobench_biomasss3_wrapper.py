import logging
import os
from collections.abc import Callable
from datetime import date
from typing import TypeAlias

import cv2
import kornia as K
import numpy as np
import rasterio
import torch
from pyproj import Transformer
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset

logging.getLogger("rasterio").setLevel(logging.ERROR)

Path: TypeAlias = str | os.PathLike[str]


class CoBenchBiomassS3(NonGeoDataset):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l3_biomass_s3/biomass_s3.zip"
    splits = ("train", "test", "val")
    all_band_names = (
        "Oa01_radiance",
        "Oa02_radiance",
        "Oa03_radiance",
        "Oa04_radiance",
        "Oa05_radiance",
        "Oa06_radiance",
        "Oa07_radiance",
        "Oa08_radiance",
        "Oa09_radiance",
        "Oa10_radiance",
        "Oa11_radiance",
        "Oa12_radiance",
        "Oa13_radiance",
        "Oa14_radiance",
        "Oa15_radiance",
        "Oa16_radiance",
        "Oa17_radiance",
        "Oa18_radiance",
        "Oa19_radiance",
        "Oa20_radiance",
        "Oa21_radiance",
    )
    all_band_scale = (
        0.0139465,
        0.0133873,
        0.0121481,
        0.0115198,
        0.0100953,
        0.0123538,
        0.00879161,
        0.00876539,
        0.0095103,
        0.00773378,
        0.00675523,
        0.0071996,
        0.00749684,
        0.0086512,
        0.00526779,
        0.00530267,
        0.00493004,
        0.00549962,
        0.00502847,
        0.00326378,
        0.00324118,
    )
    # stats for training set
    biomass_mean = 92.3196 #93.8317
    biomass_std = 117.8162 #110.5369

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        mode: str = "static",  # or 'series'
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.download = download
        self.mode = mode

        assert split in ["train", "test", "val"]
        self.split = split

        self.img_dir = os.path.join(root, split, "s3_olci")
        self.biomass_dir = os.path.join(root, split, "biomass")

        self.pids = os.listdir(self.biomass_dir)

        if self.mode == "static":
            self.static_csv = os.path.join(root, split, "static_fnames.csv")
            with open(self.static_csv, "r") as f:
                lines = f.readlines()
                self.static_img = {}
                for line in lines:
                    dirname = line.strip().split(",")[0]
                    img_fname = line.strip().split(",")[1]
                    self.static_img[dirname] = img_fname

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (16 * 0.3) ** 2  # patchsize 16 pix, gsd 300m

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        images, meta_infos = self._load_image(index)
        biomass = self._load_target(index)

        if self.mode == "static":
            sample = {"image": images[0], "groundtruth": biomass, "meta": meta_infos[0]}
        elif self.mode == "series":
            sample = {"image": images, "groundtruth": biomass, "meta": meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index):
        pid = self.pids[index]
        s3_path = os.path.join(self.img_dir, pid.replace(".tif", ""))

        if self.mode == "static":
            img_fname = self.static_img[pid.replace(".tif", "")]
            s3_paths = [os.path.join(s3_path, img_fname)]
        else:
            img_fnames = os.listdir(s3_path)
            s3_paths = []
            for img_fname in img_fnames:
                s3_paths.append(os.path.join(s3_path, img_fname))

        imgs = []
        meta_infos = []
        for img_path in s3_paths:
            with rasterio.open(img_path) as src:
                img = src.read()
                chs = []
                for b in range(21):
                    ch = cv2.resize(img[b], (282, 282), interpolation=cv2.INTER_CUBIC)
                    ch = ch * self.all_band_scale[b]
                    chs.append(ch)
                img = np.stack(chs)
                img[np.isnan(img)] = 0
                img = torch.from_numpy(img).float()

                # get lon, lat
                cx, cy = src.xy(src.height // 2, src.width // 2)
                if src.crs.to_string() != "EPSG:4326":
                    # convert to lon, lat
                    crs_transformer = Transformer.from_crs(src.crs, "epsg:4326", always_xy=True)
                    lon, lat = crs_transformer.transform(cx, cy)
                else:
                    lon, lat = cx, cy
                # get time
                img_fname = os.path.basename(img_path)
                date_str = img_fname.split("_")[1][:8]
                date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                delta = (date_obj - self.reference_date).days
                meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
                meta_info = torch.from_numpy(meta_info)

            imgs.append(img)
            meta_infos.append(meta_info)

        if self.mode == "series":
            # pad to 4 images if less than 4
            while len(imgs) < 4:
                imgs.append(img)
                meta_infos.append(meta_info)

        return imgs, meta_infos

    def _load_target(self, index):
        pid = self.pids[index]
        biomass_path = os.path.join(self.biomass_dir, pid)

        with rasterio.open(biomass_path) as src:
            biomass = src.read(1)
            biomass = cv2.resize(biomass, (282, 282), interpolation=cv2.INTER_CUBIC)
            #biomass[np.isnan(biomass)] = 0
            #biomass = (biomass - self.biomass_mean) / self.biomass_std # normalize target
            biomass = torch.from_numpy(biomass.astype("float32"))

        return biomass


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
    def forward(self, batch: dict[str, Tensor]):
        x, target = batch["image"], batch["groundtruth"]
        x = self.norm(x)
        if self.norm_target:
            target = self.norm_target(target)
            target = target.squeeze(0)
        x_out, target_out = self.transform(x, target.unsqueeze(0))
        return x_out.squeeze(0), target_out.squeeze(0), batch["meta"]


class CoBenchBiomassS3Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.mode = config.mode
        self.band_stats = config.band_stats
        self.target_stats = config.target_stats

    def create_dataset(self):
        train_transform = RegDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats, target_stats=self.target_stats)
        eval_transform = RegDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats, target_stats=self.target_stats)

        dataset_train = CoBenchBiomassS3(
            root=self.root_dir, split="train", mode=self.mode, transforms=train_transform
        )
        dataset_test = CoBenchBiomassS3(
            root=self.root_dir, split="test", mode=self.mode, transforms=eval_transform
        )
        dataset_val = CoBenchBiomassS3(
            root=self.root_dir, split="val", mode=self.mode, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
