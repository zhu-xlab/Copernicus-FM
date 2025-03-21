import kornia.augmentation as K
import torch
from torchgeo.datasets import So2Sat
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
from pyproj import Transformer
import h5py
from typing import TypeAlias, ClassVar
import pathlib
Path: TypeAlias = str | os.PathLike[str]

class CoBenchLCZS12(So2Sat):

    versions = ('cobench')
    filenames_by_version: ClassVar[dict[str, dict[str, str]]] = {
        'cobench': {
            'train': 'lcz_train.h5', # https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l3_lcz_s2/lcz_train.h5
            'val': 'lcz_val.h5', # https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l3_lcz_s2/lcz_val.h5
            'test': 'lcz_test.h5' # https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l3_lcz_s2/lcz_test.h5
        }
    }

    classes = (
        'Compact high rise',
        'Compact mid rise',
        'Compact low rise',
        'Open high rise',
        'Open mid rise',
        'Open low rise',
        'Lightweight low rise',
        'Large low rise',
        'Sparsely built',
        'Heavy industry',
        'Dense trees',
        'Scattered trees',
        'Bush, scrub',
        'Low plants',
        'Bare rock or paved',
        'Bare soil or sand',
        'Water',
    )

    all_s1_band_names = (
        'S1_B1', # VH real
        'S1_B2', # VH imaginary
        'S1_B3', # VV real
        'S1_B4', # VV imaginary
        'S1_B5', # VH intensity
        'S1_B6', # VV intensity
        'S1_B7', # PolSAR covariance matrix off-diagonal real
        'S1_B8', # PolSAR covariance matrix off-diagonal imaginary
    )
    all_s2_band_names = (
        'S2_B02',
        'S2_B03',
        'S2_B04',
        'S2_B05',
        'S2_B06',
        'S2_B07',
        'S2_B08',
        'S2_B8A',
        'S2_B11',
        'S2_B12',
    )
    all_band_names = all_s1_band_names + all_s2_band_names

    rgb_bands = ('S2_B04', 'S2_B03', 'S2_B02')

    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        's1': all_s1_band_names,
        's2': all_s2_band_names,
        'rgb': rgb_bands,
    }

    def __init__(
        self,
        root: Path = 'data',
        version: str = 'cobench', # only supported version
        split: str = 'train',
        modality: str = 's2',
        bands: Sequence[str] = BAND_SETS['s2'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        assert version in self.versions
        assert split in self.filenames_by_version[version]

        self._validate_bands(bands)
        self.s1_band_indices: np.typing.NDArray[np.int_] = np.array(
            [
                self.all_s1_band_names.index(b)
                for b in bands
                if b in self.all_s1_band_names
            ]
        ).astype(int)

        self.s1_band_names = [self.all_s1_band_names[i] for i in self.s1_band_indices]

        self.s2_band_indices: np.typing.NDArray[np.int_] = np.array(
            [
                self.all_s2_band_names.index(b)
                for b in bands
                if b in self.all_s2_band_names
            ]
        ).astype(int)

        self.s2_band_names = [self.all_s2_band_names[i] for i in self.s2_band_indices]

        self.modality = modality
        self.bands = bands

        self.root = root
        self.version = version
        self.split = split
        self.transforms = transforms
        # self.checksum = checksum

        self.fn = os.path.join(self.root, self.filenames_by_version[version][split])

        # if not self._check_integrity():
        #     raise DatasetNotFoundError(self)

        with h5py.File(self.fn, 'r') as f:
            self.size: int = f['label'].shape[0]

        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """

        with h5py.File(self.fn, 'r') as f:
            if self.modality == 's1':
                s1 = f['sen1'][index].astype(np.float32)
                s1 = np.take(s1, indices=self.s1_band_indices, axis=2)
                s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
                s1 = torch.from_numpy(s1)
                image = s1
            elif self.modality == 's2':
                s2 = f['sen2'][index].astype(np.float32)
                s2 = np.take(s2, indices=self.s2_band_indices, axis=2)
                s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format
                s2 = torch.from_numpy(s2)
                image = s2
            # convert one-hot encoding to int64 then torch int
            label = torch.tensor(f['label'][index].argmax())

        meta_info = np.array([np.nan, np.nan, np.nan, self.patch_area]).astype(np.float32)
        
        sample = {'image': image, 'label': label, 'meta': torch.from_numpy(meta_info)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class ClsDataAugmentation(torch.nn.Module):

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

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                #K.RandomResizedCrop(size=size, scale=(0.8,1.0)),
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


class ClsDataAugmentationSoftCon(torch.nn.Module):

    def __init__(self, split, size, band_stats):
        super().__init__()

        if band_stats is not None:
            self.mean = band_stats['mean']
            self.std = band_stats['std']
        else:
            self.mean = [0.0]
            self.std = [1.0]

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
        sample_img = sample["image"]
        img_bands = []
        for b in range(10):
            img = sample_img[b,:,:].clone()
            ## normalize
            img = self.normalize(img,self.mean[b],self.std[b])         
            img_bands.append(img)
            if b==0:
                # pad zero to B01, B09, B10
                img_bands.insert(b,torch.zeros_like(img))
            if b==7:
                img_bands.insert(b,torch.zeros_like(img))
                img_bands.insert(b,torch.zeros_like(img))
            
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


class CoBenchLCZS12Dataset:
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
            print('Norm with softcon.')
            train_transform = ClsDataAugmentationSoftCon(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = ClsDataAugmentationSoftCon(split="test", size=self.img_size, band_stats=self.band_stats)
        else:
            train_transform = ClsDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = ClsDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = CoBenchLCZS12(
            root=self.root_dir, split="train", modality=self.modality, bands=self.bands, transforms=train_transform
        )
        dataset_val = CoBenchLCZS12(
            root=self.root_dir, split="val", modality=self.modality, bands=self.bands, transforms=eval_transform
        )
        dataset_test = CoBenchLCZS12(
            root=self.root_dir, split="test", modality=self.modality, bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test