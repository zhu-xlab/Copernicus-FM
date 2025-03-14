import os
import rasterio
import json
import gzip
from tqdm import tqdm
import shutil

json_path = 'data/example_100_grids_cleaned/fnames_sampled.json.gz'
root_dir = 'data/example_100_grids_cleaned/'
#json_path = 'raw_data/fnames_union.json.gz'
#root_dir = 'raw_data/'


with gzip.open(json_path, 'rt', encoding='utf-8') as gz_file:
    fnames_json = json.load(gz_file)
    print(len(fnames_json))

# check if the files are corrupted
# option 2: check the file size > 1kb
count_s1 = 0
count_s2 = 0
count_s3 = 0
count_s5p_co = 0
count_s5p_no2 = 0
count_s5p_so2 = 0
count_s5p_o3 = 0
count_dem = 0

count_s1_grid = 0
count_s2_grid = 0
count_s3_grid = 0
count_s5p_co_grid = 0
count_s5p_no2_grid = 0
count_s5p_so2_grid = 0
count_s5p_o3_grid = 0
count_dem_grid = 0

count_s1_grid_complete = 0
count_s2_grid_complete = 0
count_s3_grid_complete = 0





for grid_id in tqdm(fnames_json.keys()):
    # s1
    s1_local_grids = fnames_json[grid_id]['s1_grd']
    if s1_local_grids:
        len_grids = len(s1_local_grids)
        grid_count = 0
        grid_count_complete = 0
        for local_grid_coord in s1_local_grids.keys():
            local_grid_paths = s1_local_grids[local_grid_coord]
            local_count = 0
            for local_img_path in local_grid_paths:
                # option 2: check the file size > 1kb
                if os.path.getsize(root_dir+local_img_path) < 1000:
                    #print('corrupted', local_img_path)
                    count_s1 += 1
                    local_count += 1
                    # remove the image
                    os.remove(root_dir+local_img_path)
            if local_count == len(local_grid_paths):
                # remove the local grid
                shutil.rmtree(os.path.join(root_dir, 's1_grd_utm/images', fnames_json[grid_id]['grid_id_coord'], local_grid_coord))
                grid_count += 1
            if local_count > 0:
                grid_count_complete += 1
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's1_grd_utm/images', fnames_json[grid_id]['grid_id_coord']))
            count_s1_grid += 1
        if grid_count_complete == len_grids:
            count_s1_grid_complete += 1
        

    # s2
    s2_local_grids = fnames_json[grid_id]['s2_toa']
    if s2_local_grids:
        len_grids = len(s2_local_grids)
        grid_count = 0
        grid_count_complete = 0
        for local_grid_coord in s2_local_grids.keys():
            local_grid_paths = s2_local_grids[local_grid_coord]
            local_count = 0
            for local_img_path in local_grid_paths:
                if os.path.getsize(root_dir+local_img_path) < 1000:
                    #print('corrupted', local_img_path)
                    count_s2 += 1
                    local_count += 1
                    # remove the image
                    os.remove(root_dir+local_img_path)
            if local_count == len(local_grid_paths):
                # remove the local grid
                shutil.rmtree(os.path.join(root_dir, 's2_toa_mix/images', fnames_json[grid_id]['grid_id_coord'], local_grid_coord))
                grid_count += 1
            if local_count > 0:
                grid_count_complete += 1
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's2_toa_mix/images', fnames_json[grid_id]['grid_id_coord']))
            count_s2_grid += 1
        if grid_count_complete == len_grids:
            count_s2_grid_complete += 1
    
    # s3
    s3_fpaths = fnames_json[grid_id]['s3_olci'] # list
    if s3_fpaths:
        len_grids = len(s3_fpaths)
        grid_count = 0
        for img_path in s3_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_s3 += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's3_olci_utm/images', fnames_json[grid_id]['grid_id_coord']))
            count_s3_grid += 1
        if grid_count>0:
            count_s3_grid_complete += 1

    # s5p_co
    s5p_co_fpaths = fnames_json[grid_id]['s5p_co'] # list
    if s5p_co_fpaths:
        len_grids = len(s5p_co_fpaths)
        grid_count = 0
        for img_path in s5p_co_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_s5p_co += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's5p_l3_wgs/images', 'CO_column_number_density', fnames_json[grid_id]['grid_id_coord']))
            count_s5p_co_grid += 1

    # s5p_no2
    s5p_no2_fpaths = fnames_json[grid_id]['s5p_no2'] # list
    if s5p_no2_fpaths:
        len_grids = len(s5p_no2_fpaths)
        grid_count = 0
        for img_path in s5p_no2_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_s5p_no2 += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's5p_l3_wgs/images', 'tropospheric_NO2_column_number_density', fnames_json[grid_id]['grid_id_coord']))
            count_s5p_no2_grid += 1

    # s5p_so2
    s5p_so2_fpaths = fnames_json[grid_id]['s5p_so2'] # list
    if s5p_so2_fpaths:
        len_grids = len(s5p_so2_fpaths)
        grid_count = 0
        for img_path in s5p_so2_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_s5p_so2 += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's5p_l3_wgs/images', 'SO2_column_number_density', fnames_json[grid_id]['grid_id_coord']))
            count_s5p_so2_grid += 1

    # s5p_o3
    s5p_o3_fpaths = fnames_json[grid_id]['s5p_o3'] # list
    if s5p_o3_fpaths:
        len_grids = len(s5p_o3_fpaths)
        grid_count = 0
        for img_path in s5p_o3_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_s5p_o3 += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 's5p_l3_wgs/images', 'O3_column_number_density', fnames_json[grid_id]['grid_id_coord']))
            count_s5p_o3_grid += 1

    # dem
    dem_fpaths = fnames_json[grid_id]['dem'] # list
    if dem_fpaths:
        len_grids = len(dem_fpaths)
        grid_count = 0
        for img_path in dem_fpaths:
            if os.path.getsize(root_dir+img_path) < 1000:
                #print('corrupted', img_path)
                count_dem += 1
                grid_count += 1
                # remove the image
                os.remove(root_dir+img_path)
        if grid_count == len_grids:
            # remove the grid
            shutil.rmtree(os.path.join(root_dir, 'dem_grid_wgs/images', fnames_json[grid_id]['grid_id_coord']))
            count_dem_grid += 1

print('s1:', count_s1, 's2:', count_s2, 's3:', count_s3, 's5p_co:', count_s5p_co, 's5p_no2:', count_s5p_no2, 's5p_so2:', count_s5p_so2, 's5p_o3:', count_s5p_o3, 'dem:', count_dem)

print('s1 grid:', count_s1_grid, 's2 grid:', count_s2_grid, 's3 grid:', count_s3_grid, 's5p_co grid:', count_s5p_co_grid, 's5p_no2 grid:', count_s5p_no2_grid, 's5p_so2 grid:', count_s5p_so2_grid, 's5p_o3 grid:', count_s5p_o3_grid, 'dem grid:', count_dem_grid)

print('s1 grid complete:', count_s1_grid_complete, 's2 grid complete:', count_s2_grid_complete, 's3 grid complete:', count_s3_grid_complete)
