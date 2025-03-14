import torch
import kornia
import numpy as np
from kornia.augmentation import AugmentationSequential
from torch.utils.data import Dataset, DataLoader
from copernicuspretrain_dataset_geotiff import CopernicusPretrain
import time
import tarfile
import webdataset as wds
import gzip
import os
import shutil
import random
from tqdm import tqdm
import argparse
import itertools
from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--fnames-path', type=str, default='data_loading/fnames.json.gz')
parser.add_argument('--root-dir', type=str, default='data_loading/data')
parser.add_argument('--out-dir', type=str, default='data_loading/data_webdataset')
parser.add_argument('--maxsize-gb', type=float, default=10)
parser.add_argument('--maxcount', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--start-shard', type=int, default=0)
args = parser.parse_args()

transform_s1 = AugmentationSequential(
    kornia.augmentation.SmallestMaxSize(264),
    kornia.augmentation.CenterCrop(264),
)
transform_s2 = AugmentationSequential(
    kornia.augmentation.SmallestMaxSize(264),
    kornia.augmentation.CenterCrop(264),
)
transform_s3 = AugmentationSequential(
    kornia.augmentation.SmallestMaxSize(96),
    kornia.augmentation.CenterCrop(96),
)
transform_s5p = AugmentationSequential(
    kornia.augmentation.SmallestMaxSize(28),
    kornia.augmentation.CenterCrop(28),
)
transform_dem = AugmentationSequential(
    kornia.augmentation.SmallestMaxSize(960),
    kornia.augmentation.CenterCrop(960),
)

dataset = CopernicusPretrain(args.fnames_path, args.root_dir, transform_s1=transform_s1, transform_s2=transform_s2, transform_s3=transform_s3, transform_s5p=transform_s5p, transform_dem=transform_dem)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) # batch size can only be 1 because of varying number of images per grid


# if os.path.exists(args.out_dir):
#     print('Dir exists, please check.')
#     exit()
# os.makedirs(args.out_dir, exist_ok=True)

pattern = os.path.join(args.out_dir, f"example-%06d.tar")
maxsize = args.maxsize_gb * 1e9 # max size of each shard
maxcount = args.maxcount # samples/grids per shard

#resume_index = 4500

#with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:
with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount), start_shard=args.start_shard) as sink:
    sample = {}
    meta = {}

    for i, (sample_img, meta_data) in enumerate(tqdm(dataloader)):
        
        # s1
        meta_s1 = meta_data['s1_grd'] # num_localx4? list of string
        img_s1 = sample_img['s1_grd'] # num_localx4? list of tensor Bx2x264x264
        if len(meta_s1) > 0:
            meta_s1_new = meta_s1
            img_s1_new = img_s1
            img_s1_grid = []
            for j in range(len(meta_s1)): # num_local
                n_season = len(meta_s1[j])
                if n_season < 4:
                    pad_ids = random.choices(range(n_season), k=(4-n_season))
                    for pad_id in pad_ids:
                        meta_s1_new[j].append(meta_s1[j][pad_id])
                        img_s1_new[j].append(img_s1[j][pad_id])
                img_s1_local = torch.cat(img_s1_new[j], dim=0)
                img_s1_grid.append(img_s1_local)
            img_s1_grid = torch.stack(img_s1_grid, dim=0)
            sample['s1_grd.pth'] = img_s1_grid
            meta['s1_grd'] = meta_s1_new

        # s2
        meta_s2 = meta_data['s2_toa']
        img_s2 = sample_img['s2_toa']
        if len(meta_s2) > 0:
            meta_s2_new = meta_s2
            img_s2_new = img_s2
            img_s2_grid = []
            for j in range(len(meta_s2)):
                n_season = len(meta_s2[j])
                if n_season < 4:
                    pad_ids = random.choices(range(n_season), k=(4-n_season))
                    for pad_id in pad_ids:
                        meta_s2_new[j].append(meta_s2[j][pad_id])
                        img_s2_new[j].append(img_s2[j][pad_id])
                img_s2_local = torch.cat(img_s2_new[j], dim=0)
                img_s2_grid.append(img_s2_local)
            img_s2_grid = torch.stack(img_s2_grid, dim=0)
            sample['s2_toa.pth'] = img_s2_grid
            meta['s2_toa'] = meta_s2_new
        
        # s3
        meta_s3 = meta_data['s3_olci']
        img_s3 = sample_img['s3_olci']
        if len(meta_s3) > 0:
            meta_s3_new = meta_s3
            img_s3_new = img_s3
            n_season = len(meta_s3)
            if n_season < 8:
                pad_ids = random.choices(range(n_season), k=(8-n_season))
                for pad_id in pad_ids:
                    meta_s3_new.append(meta_s3[pad_id])
                    img_s3_new.append(img_s3[pad_id])
            img_s3_grid = torch.cat(img_s3_new, dim=0)
            sample['s3_olci.pth'] = img_s3_grid
            meta['s3_olci'] = meta_s3_new
        
        # s5p
        meta_s5p_co = meta_data['s5p_co']
        img_s5p_co = sample_img['s5p_co']
        if len(meta_s5p_co) > 0:
            meta_s5p_co_new = meta_s5p_co
            img_s5p_co_new = img_s5p_co
            n_season = len(meta_s5p_co)
            if n_season < 12:
                pad_ids = random.choices(range(n_season), k=(12-n_season))
                for pad_id in pad_ids:
                    meta_s5p_co_new.append(meta_s5p_co[pad_id])
                    img_s5p_co_new.append(img_s5p_co[pad_id])
            img_s5p_co_grid = torch.cat(img_s5p_co_new, dim=0)
            sample['s5p_co.pth'] = img_s5p_co_grid
            meta['s5p_co'] = meta_s5p_co_new
        
        meta_s5p_no2 = meta_data['s5p_no2']
        img_s5p_no2 = sample_img['s5p_no2']
        if len(meta_s5p_no2) > 0:
            meta_s5p_no2_new = meta_s5p_no2
            img_s5p_no2_new = img_s5p_no2
            n_season = len(meta_s5p_no2)
            if n_season < 12:
                pad_ids = random.choices(range(n_season), k=(12-n_season))
                for pad_id in pad_ids:
                    meta_s5p_no2_new.append(meta_s5p_no2[pad_id])
                    img_s5p_no2_new.append(img_s5p_no2[pad_id])
            img_s5p_no2_grid = torch.cat(img_s5p_no2_new, dim=0)
            sample['s5p_no2.pth'] = img_s5p_no2_grid
            meta['s5p_no2'] = meta_s5p_no2_new
        
        meta_s5p_o3 = meta_data['s5p_o3']
        img_s5p_o3 = sample_img['s5p_o3']
        if len(meta_s5p_o3) > 0:
            meta_s5p_o3_new = meta_s5p_o3
            img_s5p_o3_new = img_s5p_o3
            n_season = len(meta_s5p_o3)
            if n_season < 12:
                pad_ids = random.choices(range(n_season), k=(12-n_season))
                for pad_id in pad_ids:
                    meta_s5p_o3_new.append(meta_s5p_o3[pad_id])
                    img_s5p_o3_new.append(img_s5p_o3[pad_id])
            img_s5p_o3_grid = torch.cat(img_s5p_o3_new, dim=0)
            sample['s5p_o3.pth'] = img_s5p_o3_grid
            meta['s5p_o3'] = meta_s5p_o3_new
        
        meta_s5p_so2 = meta_data['s5p_so2']
        img_s5p_so2 = sample_img['s5p_so2']
        if len(meta_s5p_so2) > 0:
            meta_s5p_so2_new = meta_s5p_so2
            img_s5p_so2_new = img_s5p_so2
            n_season = len(meta_s5p_so2)
            if n_season < 12:
                pad_ids = random.choices(range(n_season), k=(12-n_season))
                for pad_id in pad_ids:
                    meta_s5p_so2_new.append(meta_s5p_so2[pad_id])
                    img_s5p_so2_new.append(img_s5p_so2[pad_id])
            img_s5p_so2_grid = torch.cat(img_s5p_so2_new, dim=0)
            sample['s5p_so2.pth'] = img_s5p_so2_grid
            meta['s5p_so2'] = meta_s5p_so2_new
        
        # dem
        meta_dem = meta_data['dem']
        img_dem = sample_img['dem']
        if len(meta_dem) > 0:
            img_dem_grid = img_dem[0][0]
            sample['dem.pth'] = img_dem_grid
            meta['dem'] = meta_dem


        xkey = meta['dem'][0][0].split('_')[0] # grid id
        sample["__key__"] = xkey
        sample["json"] = meta


        sink.write(sample)
        #break

print('Done.')
