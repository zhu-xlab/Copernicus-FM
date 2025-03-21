import kornia.augmentation as K
import torch
from torchgeo.datasets import EuroSAT
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
from pyproj import Transformer
from typing import TypeAlias, ClassVar
import pathlib
Path: TypeAlias = str | os.PathLike[str]

class CoBenchEuroSATS2(EuroSAT):
    url = "https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_eurosat_s1s2/eurosat_s2.zip"
    base_dir = 'all_imgs'
    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'eurosat-train.txt',
        'val': 'eurosat-val.txt',
        'test': 'eurosat-test.txt',
    }
    all_band_names = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B09',
        'B10',
        'B11',
        'B12',
        'B8A',
    )
    rgb_bands = ('B04', 'B03', 'B02')
    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        'rgb': rgb_bands,
        'all-ssl4eo': ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['all-ssl4eo'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        #self._verify()

        self.valid_fns = []
        self.classes = []
        with open(os.path.join(self.root, self.split_filenames[split])) as f:
            for fn in f:
                self.valid_fns.append(fn.strip().replace('.jpg', '.tif'))
                cls_name = fn.strip().split('_')[0]
                if cls_name not in self.classes:
                    self.classes.append(cls_name)
        self.classes = sorted(self.classes)

        self.root = os.path.join(self.root, self.base_dir)
        #root_path = pathlib.Path(root,split)
        #self.classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.patch_area = (16*10/1000)**2 # patchsize 16 pix, gsd 10m

    def __len__(self):
        return len(self.valid_fns)

    def __getitem__(self, index):

        image, coord, label = self._load_image(index)
        meta_info = np.array([coord[0], coord[1], np.nan, self.patch_area]).astype(np.float32)
        sample = {'image': image, 'label': label, 'meta': torch.from_numpy(meta_info)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        fname = self.valid_fns[index]
        dirname = fname.split('_')[0]
        img_path = os.path.join(self.root, dirname, fname)
        target = self.class_to_idx[dirname]

        with rasterio.open(img_path) as src:
            image = src.read(self.band_indices).astype('float32')
            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy

        return torch.from_numpy(image), (lon,lat), target


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
        for b in range(13):
            img = sample_img[b,:,:].clone()
            ## normalize
            img = self.normalize(img,self.mean[b],self.std[b])         
            img_bands.append(img)
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



class CoBenchEuroSATS2Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.band_names
        self.band_stats = config.band_stats
        self.norm_form = config.norm_form if 'norm_form' in config else None

    def create_dataset(self):
        if self.norm_form == 'softcon':
            train_transform = ClsDataAugmentationSoftCon(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = ClsDataAugmentationSoftCon(split="test", size=self.img_size, band_stats=self.band_stats)
        else:
            train_transform = ClsDataAugmentation(split="train", size=self.img_size, band_stats=self.band_stats)
            eval_transform = ClsDataAugmentation(split="test", size=self.img_size, band_stats=self.band_stats)

        dataset_train = CoBenchEuroSATS2(
            root=self.root_dir, split="train", bands=self.bands, transforms=train_transform
        )
        dataset_val = CoBenchEuroSATS2(
            root=self.root_dir, split="val", bands=self.bands, transforms=eval_transform
        )
        dataset_test = CoBenchEuroSATS2(
            root=self.root_dir, split="test", bands=self.bands, transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test