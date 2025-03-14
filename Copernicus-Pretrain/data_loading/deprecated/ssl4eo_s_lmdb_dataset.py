import lmdb
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv


class SSL4EO_S_lmdb(Dataset):
    def __init__(self, lmdb_path, key_path, slurm_job=False, mode=['s1_grd','s2_toa','s3_olci','s5p_co','s5p_no2','s5p_so2','s5p_o3','dem'], s1_transform=None, s2_transform=None, s3_transform=None, s5p_transform=None, dem_transform=None):
        self.lmdb_path = lmdb_path
        self.key_path = key_path
        self.slurm_job = slurm_job
        self.mode = mode
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.s3_transform = s3_transform
        self.s5p_transform = s5p_transform
        self.dem_transform = dem_transform

        if not self.slurm_job:
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        #self.txn = self.env.begin(write=False) # Q: when to close the txn? # 
        self.keys = {}
        with open(key_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                modality, meta_info = row[0], row[1]
                if modality=='s1_grd' or modality=='s2_toa':
                    _, grid_id, local_grid_id, date = meta_info.split('/')
                    if grid_id not in self.keys:
                        self.keys[grid_id] = {}
                    if modality not in self.keys[grid_id]:
                        self.keys[grid_id][modality] = {}
                    if local_grid_id not in self.keys[grid_id][modality]:
                        self.keys[grid_id][modality][local_grid_id] = []
                    self.keys[grid_id][modality][local_grid_id].append(meta_info)
                elif modality=='s3_olci' or modality=='s5p_co' or modality=='s5p_no2' or modality=='s5p_so2' or modality=='s5p_o3':
                    _, grid_id, date = meta_info.split('/')
                    if grid_id not in self.keys:
                        self.keys[grid_id] = {}
                    if modality not in self.keys[grid_id]:
                        self.keys[grid_id][modality] = []
                    self.keys[grid_id][modality].append(meta_info)
                elif modality=='dem':
                    _, grid_id = meta_info.split('/')
                    if grid_id not in self.keys:
                        self.keys[grid_id] = {}
                    if modality not in self.keys[grid_id]:
                        self.keys[grid_id][modality] = []
                    self.keys[grid_id][modality].append(meta_info)
        self.indices = list(self.keys.keys())

    def __len__(self):
        return len(self.indices)

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, idx): 
        if self.slurm_job:
            # Delay loading LMDB data until after initialization
            if self.env is None:
                self._init_db()
        # get all images in a random local grid in one era5 grid (for batch loading)
        grid_id = self.indices[idx]
        grid_keys = self.keys[grid_id]
        sample = {}
        meta_info = {}

        with self.env.begin(write=False) as txn:
            # s1
            if 's1_grd' in self.mode:
                sample['s1_grd'] = []
                meta_info['s1_grd'] = []
                if 's1_grd' in grid_keys:
                    local_grids = list(grid_keys['s1_grd'].keys()) # list of local grid ids
                    for local_grid_id in local_grids:
                        local_keys = grid_keys['s1_grd'][local_grid_id] # list of 4 keys
                        local_meta_info = []
                        local_imgs = []
                        for key in local_keys:
                            #print(key)
                            img_bytes = txn.get(key.encode('utf-8'))
                            img = np.frombuffer(img_bytes, dtype=np.float32).reshape(264, 264, 2)
                            if self.s1_transform:
                                img = self.s1_transform(img)
                            local_meta_info.append(key)
                            local_imgs.append(img)
                        ## pad time stamps to 4
                        #if len(s1_meta_info) < 4:
                        #    s1_meta_info += [s1_meta_info[-1]] * (4 - len(s1_meta_info))
                        #    s1_imgs += [s1_imgs[-1]] * (4 - len(s1_imgs))
                        sample['s1_grd'].append(local_imgs)
                        meta_info['s1_grd'].append(local_meta_info)

            # s2
            if 's2_toa' in self.mode:
                sample['s2_toa'] = []
                meta_info['s2_toa'] = []
                if 's2_toa' in grid_keys:
                    local_grids = list(grid_keys['s2_toa'].keys())
                    for local_grid_id in local_grids:
                        local_keys = grid_keys['s2_toa'][local_grid_id]
                        local_meta_info = []
                        local_imgs = []
                        for key in local_keys:
                            img_bytes = txn.get(key.encode('utf-8'))
                            img = np.frombuffer(img_bytes, dtype=np.int16).reshape(264, 264, 13)
                            if self.s2_transform:
                                img = self.s2_transform(img)
                            local_meta_info.append(key)
                            local_imgs.append(img)
                        sample['s2_toa'].append(local_imgs)
                        meta_info['s2_toa'].append(local_meta_info)

            # s3
            if 's3_olci' in self.mode:
                sample['s3_olci'] = []
                meta_info['s3_olci'] = []
                if 's3_olci' in grid_keys:
                    local_keys = grid_keys['s3_olci']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(96, 96, 21)
                        if self.s3_transform:
                            img = self.s3_transform(img)
                        meta_info['s3_olci'].append(key)
                        sample['s3_olci'].append(img)

            # s5p
            if 's5p_co' in self.mode:
                sample['s5p_co'] = []
                meta_info['s5p_co'] = []
                if 's5p_co' in grid_keys:
                    local_keys = grid_keys['s5p_co']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(28, 28, 1)
                        if self.s5p_transform:
                            img = self.s5p_transform(img)
                        meta_info['s5p_co'].append(key)
                        sample['s5p_co'].append(img)

            if 's5p_no2' in self.mode:
                sample['s5p_no2'] = []
                meta_info['s5p_no2'] = []
                if 's5p_no2' in grid_keys:
                    local_keys = grid_keys['s5p_no2']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(28, 28, 1)
                        if self.s5p_transform:
                            img = self.s5p_transform(img)
                        meta_info['s5p_no2'].append(key)
                        sample['s5p_no2'].append(img)

            if 's5p_so2' in self.mode:
                sample['s5p_so2'] = []
                meta_info['s5p_so2'] = []
                if 's5p_so2' in grid_keys:
                    local_keys = grid_keys['s5p_so2']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(28, 28, 1)
                        if self.s5p_transform:
                            img = self.s5p_transform(img)
                        meta_info['s5p_so2'].append(key)
                        sample['s5p_so2'].append(img)

            if 's5p_o3' in self.mode:
                sample['s5p_o3'] = []
                meta_info['s5p_o3'] = []
                if 's5p_o3' in grid_keys:
                    local_keys = grid_keys['s5p_o3']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(28, 28, 1)
                        if self.s5p_transform:
                            img = self.s5p_transform(img)
                        meta_info['s5p_o3'].append(key)
                        sample['s5p_o3'].append(img)
            
            # dem
            if 'dem' in self.mode:
                sample['dem'] = []
                meta_info['dem'] = []
                if 'dem' in grid_keys:
                    local_keys = grid_keys['dem']
                    for key in local_keys:
                        img_bytes = txn.get(key.encode('utf-8'))
                        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(960,960,1)
                        if self.dem_transform:
                            img = self.dem_transform(img)
                        meta_info['dem'].append(key)
                        sample['dem'].append(img)



        return sample, meta_info