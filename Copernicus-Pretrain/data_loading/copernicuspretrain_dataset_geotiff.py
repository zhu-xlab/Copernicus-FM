import os
import gzip
import json
import numpy as np
import rasterio
import re
from torch.utils.data import Dataset, DataLoader
import torch
from kornia.augmentation import AugmentationSequential
import kornia
import argparse


class CopernicusPretrain(Dataset):
    def __init__(self, fnames_path, root_dir, modality=['s1_grd', 's2_toa', 's3_olci', 's5p_co', 's5p_no2', 's5p_so2', 's5p_o3', 'dem'], transform_s1=None, transform_s2=None, transform_s3=None, transform_s5p=None, transform_dem=None):
        with gzip.open(fnames_path, 'rt', encoding='utf-8') as gz_file:
            self.fnames_json = json.load(gz_file)
        self.grid_ids = list(self.fnames_json.keys())
        self.root_dir = root_dir
        self.transform_s1 = transform_s1
        self.transform_s2 = transform_s2
        self.transform_s3 = transform_s3
        self.transform_s5p = transform_s5p
        self.transform_dem = transform_dem
        self.modality = modality

    def __len__(self):
        return len(self.grid_ids)

    def get_s1_s2(self,grid_id,modality):
        arrays = []
        meta_data = []
        local_grids = list(self.fnames_json[grid_id][modality].keys())
        grid_id_coord = self.fnames_json[grid_id]['grid_id_coord']
        for local_grid in local_grids:
            local_fpaths = self.fnames_json[grid_id][modality][local_grid]
            imgs = []
            meta = []
            for local_fpath in local_fpaths:
                try:
                    with rasterio.open(os.path.join(self.root_dir, local_fpath)) as src:
                        img = src.read()

                    if modality=='s1_grd' and self.transform_s1:
                        img = torch.from_numpy(img).unsqueeze(0)
                        img = self.transform_s1(img).squeeze(0)
                    elif modality=='s2_toa' and self.transform_s2:
                        img = torch.from_numpy(img.astype(np.int16)).unsqueeze(0)
                        img = self.transform_s2(img.to(torch.float32)).squeeze(0).to(torch.int16)
                    imgs.append(img)
                    fname = local_fpath.split('/')[-1]
                    date = re.search(r'(\d{8})T', fname).group(1)
                    meta_info = f"{grid_id_coord}/{local_grid}/{date}"
                    meta.append(meta_info)
                except:
                    print('Error file:', local_fpath)
            arrays.append(imgs)
            meta_data.append(meta)
        return arrays, meta_data

    def get_s3(self,grid_id):
        arrays = []
        meta_data = []
        fpaths = self.fnames_json[grid_id]['s3_olci']
        grid_id_coord = self.fnames_json[grid_id]['grid_id_coord']
        for fpath in fpaths:
            # change utm to wgs
            fpath = fpath.replace('s3_olci_utm','s3_olci_wgs')
            try:
                with rasterio.open(os.path.join(self.root_dir, fpath)) as src:
                    img = src.read()

                if self.transform_s3:
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = self.transform_s3(img).squeeze(0)
                arrays.append(img)
                fname = fpath.split('/')[-1]
                date = re.search(r'(\d{8})T', fname).group(1)
                meta_info = f"{grid_id_coord}/{date}"
                meta_data.append(meta_info)
            except:
                print('Error file:', fpath)
        return arrays, meta_data

    def get_s5p(self,grid_id,modality):
        arrays = []
        meta_data = []
        fpaths = self.fnames_json[grid_id][modality]
        grid_id_coord = self.fnames_json[grid_id]['grid_id_coord']
        for fpath in fpaths:
            try:
                with rasterio.open(os.path.join(self.root_dir, fpath)) as src:
                    img = src.read()

                if self.transform_s5p:
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = self.transform_s5p(img).squeeze(0)
                arrays.append(img)
                fname = fpath.split('/')[-1]
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
                date = f"{match.group(1)}{match.group(2)}{match.group(3)}"
                meta_info = f"{grid_id_coord}/{date}"
                meta_data.append(meta_info)
            except:
                print('Error file:', fpath)
        return arrays, meta_data

    def get_dem(self,grid_id):
        fpath = self.fnames_json[grid_id]['dem'][0]
        with rasterio.open(os.path.join(self.root_dir, fpath)) as src:
           img = src.read()

        if self.transform_dem:
            img = torch.from_numpy(img).unsqueeze(0)
            img = self.transform_dem(img).squeeze(0)
        return img

    def __getitem__(self, idx):
        grid_id = self.grid_ids[idx]
        grid_id_coord = self.fnames_json[grid_id]['grid_id_coord']
        sample = {}
        meta_data = {}
        # s1
        if 's1_grd' in self.modality:
            if self.fnames_json[grid_id]['s1_grd']=={}:
                arr_s1 = []
                meta_s1 = []
            else:
                arr_s1, meta_s1 = self.get_s1_s2(grid_id,'s1_grd')
            sample['s1_grd'] = arr_s1
            meta_data['s1_grd'] = meta_s1
        # s2
        if 's2_toa' in self.modality:
            if self.fnames_json[grid_id]['s2_toa']=={}:
                arr_s2 = []
                meta_s2 = []
            else:
                arr_s2, meta_s2 = self.get_s1_s2(grid_id,'s2_toa')
            sample['s2_toa'] = arr_s2
            meta_data['s2_toa'] = meta_s2
        # s3
        if 's3_olci' in self.modality:
            if self.fnames_json[grid_id]['s3_olci']==[]:
                arr_s3 = []
                meta_s3 = []
            else:
                arr_s3, meta_s3 = self.get_s3(grid_id)
            sample['s3_olci'] = arr_s3
            meta_data['s3_olci'] = meta_s3
        # s5p_co
        if 's5p_co' in self.modality:
            if self.fnames_json[grid_id]['s5p_co']==[]:
                arr_s5p_co = []
                meta_s5p_co = []
            else:
                arr_s5p_co, meta_s5p_co = self.get_s5p(grid_id,'s5p_co')
            sample['s5p_co'] = arr_s5p_co
            meta_data['s5p_co'] = meta_s5p_co
        # s5p_no2
        if 's5p_no2' in self.modality:
            if self.fnames_json[grid_id]['s5p_no2']==[]:
                arr_s5p_no2 = []
                meta_s5p_no2 = []
            else:
                arr_s5p_no2, meta_s5p_no2 = self.get_s5p(grid_id,'s5p_no2')
            sample['s5p_no2'] = arr_s5p_no2
            meta_data['s5p_no2'] = meta_s5p_no2
        # s5p_o3
        if 's5p_o3' in self.modality:
            if self.fnames_json[grid_id]['s5p_o3']==[]:
                arr_s5p_o3 = []
                meta_s5p_o3 = []
            else:
                arr_s5p_o3, meta_s5p_o3 = self.get_s5p(grid_id,'s5p_o3')
            sample['s5p_o3'] = arr_s5p_o3
            meta_data['s5p_o3'] = meta_s5p_o3
        # s5p_so2
        if 's5p_so2' in self.modality:
            if self.fnames_json[grid_id]['s5p_so2']==[]:
                arr_s5p_so2 = []
                meta_s5p_so2 = []
            else:
                arr_s5p_so2, meta_s5p_so2 = self.get_s5p(grid_id,'s5p_so2')
            sample['s5p_so2'] = arr_s5p_so2
            meta_data['s5p_so2'] = meta_s5p_so2
        # dem
        if 'dem' in self.modality:
            if self.fnames_json[grid_id]['dem']==[]:
                arr_dem = []
                meta_data['dem'] = []
            else:
                try:
                    arr_dem = self.get_dem(grid_id)
                    meta_data['dem'] = [grid_id_coord]
                except:
                    arr_dem = []
                    meta_data['dem'] = []
                    print('Error file:', self.fnames_json[grid_id]['dem'][0])
            sample['dem'] = arr_dem
            

        return sample, meta_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fnames_path', type=str, default='data_loading/fnames.json.gz')
    parser.add_argument('--root_dir', type=str, default='data_loading/data')
    args = parser.parse_args()

    transform_s1 = AugmentationSequential(
        #kornia.augmentation.SmallestMaxSize(264),
        kornia.augmentation.CenterCrop(224),
    )
    transform_s2 = AugmentationSequential(
        #kornia.augmentation.SmallestMaxSize(264),
        kornia.augmentation.CenterCrop(224),
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


    CopernicusPretrain = CopernicusPretrain(args.fnames_path, args.root_dir, transform_s1=transform_s1, transform_s2=transform_s2, transform_s3=transform_s3, transform_s5p=transform_s5p, transform_dem=transform_dem)
    dataloader = DataLoader(CopernicusPretrain, batch_size=1, shuffle=True, num_workers=0) # batch size can only be 1 because of varying number of images per grid

    for i, (sample, meta_data) in enumerate(dataloader):
        #print(i)
        print('Grid ID:', meta_data['dem'][0])
        print(sample.keys())
        print(meta_data.keys())

        
        print('### S1 GRD ###')
        print('Number of s1 local patches:', len(meta_data['s1_grd']), '  ', 'Number of time stamps for first local patch:', len(meta_data['s1_grd'][0]))
        print('Example for one image:', sample['s1_grd'][0][0].shape, meta_data['s1_grd'][0][0])
        print('### S2 TOA ###')
        print('Number of s2 local patches:', len(meta_data['s2_toa']), '  ', 'Number of time stamps for first local patch:', len(meta_data['s2_toa'][0]))
        print('Example for one image:', sample['s2_toa'][0][0].shape, meta_data['s2_toa'][0][0])
        print('### S3 OLCI ###')
        print('Number of s3 time stamps:', len(meta_data['s3_olci']))
        print('Example for one image:', sample['s3_olci'][0].shape, meta_data['s3_olci'][0])
        print('### S5P ###')
        print('Number of s5p time stamps for CO/NO2/O3/SO2:', len(meta_data['s5p_co']), len(meta_data['s5p_no2']), len(meta_data['s5p_o3']), len(meta_data['s5p_so2']))
        print('Example for one CO image:', sample['s5p_co'][0].shape, meta_data['s5p_co'][0])
        print('Example for one NO2 image:', sample['s5p_no2'][0].shape, meta_data['s5p_no2'][0])
        print('Example for one O3 image:', sample['s5p_o3'][0].shape, meta_data['s5p_o3'][0])
        print('Example for one SO2 image:', sample['s5p_so2'][0].shape, meta_data['s5p_so2'][0])
        print('### DEM ###')
        print('One DEM image for the grid:', sample['dem'].shape, meta_data['dem'][0])

        break