import torch
import kornia
import numpy as np
from kornia.augmentation import AugmentationSequential
from torch.utils.data import Dataset, DataLoader
from ssl4eo_s_dataset import SSL4EO_S
import time
import tarfile
#import webdataset as wds
import lmdb
import gzip
import os
import shutil
import random
from tqdm import tqdm
import argparse
import csv


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def make_lmdb(dataset, lmdb_file, key_file, num_workers=6,mode=['s1_grd','s2_toa']):
    loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    #env = lmdb.open(lmdb_file, map_size=1099511627776,writemap=True) # continuously write to disk
    env = lmdb.open(lmdb_file, map_size=1099511627776)
    txn = env.begin(write=True)
    for index, (sample, meta_data) in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
        #pdb.set_trace()
        if 's1_grd' in mode:
            sample_s1 = sample['s1_grd']
            meta_s1 = meta_data['s1_grd']

            for i in range(len(sample_s1)):
                for j in range(len(sample_s1[i])):
                    key_str = 's1_grd/'+meta_s1[i][j]
                    key = key_str.encode('utf-8')
                    img_bytes = sample_s1[i][j].numpy().tobytes()
                    #print(sample_s1[i][j].shape, sample_s1[i][j].dtype)
                    #obj = (sample_s1[i][j].tobytes(), sample_s1[i][j].shape, meta_s1[i][j])
                    #txn.put(str(index).encode(), pickle.dumps(obj))
                    txn.put(key, img_bytes)
                    with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s1_grd',key_str])

        if 's2_toa' in mode:
            sample_s2 = sample['s2_toa']
            meta_s2 = meta_data['s2_toa']
            for i in range(len(sample_s2)):
                for j in range(len(sample_s2[i])):
                    key_str = 's2_toa/'+meta_s2[i][j]
                    key = key_str.encode('utf-8')
                    img_bytes = sample_s2[i][j].numpy().tobytes()
                    txn.put(key, img_bytes)
                    with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s2_toa',key_str])

        if 's3_olci' in mode:
            sample_s3 = sample['s3_olci']
            meta_s3 = meta_data['s3_olci']
            for i in range(len(sample_s3)):
                key_str = 's3_olci/'+meta_s3[i]
                key = key_str.encode('utf-8')
                img_bytes = sample_s3[i].numpy().tobytes()
                txn.put(key, img_bytes)
                with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s3_olci',key_str])

        if 's5p_co' in mode:
            sample_s5p_co = sample['s5p_co']
            meta_s5p_co = meta_data['s5p_co']
            for i in range(len(sample_s5p_co)):
                key_str = 's5p_co/'+meta_s5p_co[i]
                key = key_str.encode('utf-8')
                img_bytes = sample_s5p_co[i].numpy().tobytes()
                txn.put(key, img_bytes)
                with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s5p_co',key_str])  

        if 's5p_no2' in mode:
            sample_s5p_no2 = sample['s5p_no2']
            meta_s5p_no2 = meta_data['s5p_no2']
            for i in range(len(sample_s5p_no2)):
                key_str = 's5p_no2/'+meta_s5p_no2[i]
                key = key_str.encode('utf-8')
                img_bytes = sample_s5p_no2[i].numpy().tobytes()
                txn.put(key, img_bytes)
                with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s5p_no2',key_str])

        if 's5p_o3' in mode:
            sample_s5p_o3 = sample['s5p_o3']
            meta_s5p_o3 = meta_data['s5p_o3']
            for i in range(len(sample_s5p_o3)):
                key_str = 's5p_o3/'+meta_s5p_o3[i]
                key = key_str.encode('utf-8')
                img_bytes = sample_s5p_o3[i].numpy().tobytes()
                txn.put(key, img_bytes)
                with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s5p_o3',key_str])

        if 's5p_so2' in mode:
            sample_s5p_so2 = sample['s5p_so2']
            meta_s5p_so2 = meta_data['s5p_so2']
            for i in range(len(sample_s5p_so2)):
                key_str = 's5p_so2/'+meta_s5p_so2[i]
                key = key_str.encode('utf-8')
                img_bytes = sample_s5p_so2[i].numpy().tobytes()
                txn.put(key, img_bytes)
                with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['s5p_so2',key_str])

        if 'dem' in mode:
            sample_dem = sample['dem']
            meta_dem = meta_data['dem']
            key_str = 'dem/' + meta_dem[0]
            key = key_str.encode('utf-8')
            img_bytes = sample_dem.numpy().tobytes()
            txn.put(key, img_bytes)  
            with open(key_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['dem',key_str])    

        if index % 10 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()



parser = argparse.ArgumentParser()
parser.add_argument('--fnames-path', type=str, default='data_loading/fnames.json.gz')
parser.add_argument('--root-dir', type=str, default='data_loading/data')
parser.add_argument('--out-dir', type=str, default='data_loading/data_webdataset')
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--modality', type=str, default=['s1_grd', 's2_toa', 's3_olci', 's5p_co', 's5p_no2', 's5p_so2', 's5p_o3', 'dem'], help='modality to include in the lmdb')

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

ssl4eo_s = SSL4EO_S(args.fnames_path, args.root_dir, modality=args.modality, transform_s1=transform_s1, transform_s2=transform_s2, transform_s3=transform_s3, transform_s5p=transform_s5p, transform_dem=transform_dem)
dataloader = DataLoader(ssl4eo_s, batch_size=1, shuffle=False, num_workers=10) # batch size can only be 1 because of varying number of images per grid


root_dir = args.root_dir
fnames_path = args.fnames_path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
lmdb_path = os.path.join(out_dir,'ssl4eo_s_10k.lmdb')
key_path = os.path.join(out_dir,'ssl4eo_s_10k.csv')

if os.path.exists(lmdb_path):
    print(f'{lmdb_path} exists.')
    exit()
if os.path.exists(key_path):
    print(f'{key_path} exists.')
    exit()

make_lmdb(ssl4eo_s,lmdb_path,key_path,num_workers=args.num_workers,mode=args.modality)
