import os
import json
import csv
from tqdm import tqdm
from multiprocessing.dummy import Lock, Pool
import time
import gzip

class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self.lock = Lock()

    def update(self, delta: int = 1) -> int:
        with self.lock:
            self.value += delta
            return self.value

#root_dir = './raw_data/'
root_dir = '../../data/example_100_grids/'
modalities = ['s1_grd', 's2_toa', 's3_olci', 's5p_co', 's5p_no2', 's5p_so2', 's5p_o3', 'dem']
modalities_dirs = [ 's1_grd_utm/images', 
                    's2_toa_mix/images', 
                    's3_olci_utm/images', 
                    's5p_l3_wgs/images/CO_column_number_density', 
                    's5p_l3_wgs/images/tropospheric_NO2_column_number_density', 
                    's5p_l3_wgs/images/SO2_column_number_density', 
                    's5p_l3_wgs/images/O3_column_number_density', 
                    'dem_grid_wgs/images']
#all_grid_path = 'grid_index_land_s3.csv'
all_grid_path = '../sampled_locations/grid_index_globe.csv'
#out_path = './raw_data/fnames_all.json'
out_path_all = '../../data/example_100_grids/fnames_sampled_all.json.gz'
out_path_union = '../../data/example_100_grids/fnames_sampled_union.json.gz'
num_workers = 16
counter = Counter()

fnames_json = {}

with open(all_grid_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in tqdm(reader):
        grid_id = int(row[-1])
        grid_lon = float(row[1])
        grid_lat = float(row[2])
    
        str_idx = str(f"{grid_id:07d}") 
        str_lon = str(f"{grid_lon:.2f}")
        str_lat = str(f"{grid_lat:.2f}")

        grid_dirname = str_idx+'_'+str_lon+'_'+str_lat

        fnames_json[grid_id] = {}
        fnames_json[grid_id]['grid_id_coord'] = grid_dirname
        fnames_json[grid_id]['s1_grd'] = {}
        fnames_json[grid_id]['s2_toa'] = {}
        fnames_json[grid_id]['s3_olci'] = []
        fnames_json[grid_id]['s5p_co'] = []
        fnames_json[grid_id]['s5p_no2'] = []
        fnames_json[grid_id]['s5p_so2'] = []
        fnames_json[grid_id]['s5p_o3'] = []
        fnames_json[grid_id]['dem'] = []


start_time = time.time()

def worker(idx):
    grid_id = idx
    for modality_dir in modalities_dirs:
        if 's1' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                local_ids = os.listdir(grid_dir)
                for local_id in local_ids:
                    local_dir = os.path.join(grid_dir, local_id)
                    fnames_json[grid_id]['s1_grd'][local_id] = []
                    local_fnames = os.listdir(local_dir)
                    for fname in local_fnames:
                        if fname.endswith('.tif'):
                            fpath = os.path.join(local_dir, fname)
                            # remove root_dir from fpath
                            fpath = fpath.replace(root_dir, '')
                            fnames_json[grid_id]['s1_grd'][local_id].append(fpath)
        elif 's2' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                local_ids = os.listdir(grid_dir)
                for local_id in local_ids:
                    local_dir = os.path.join(grid_dir, local_id)
                    fnames_json[grid_id]['s2_toa'][local_id] = []
                    local_fnames = os.listdir(local_dir)
                    for fname in local_fnames:
                        if fname.endswith('.tif'):
                            fpath = os.path.join(local_dir, fname)
                            fpath = fpath.replace(root_dir, '')
                            fnames_json[grid_id]['s2_toa'][local_id].append(fpath)
        elif 's3_olci' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['s3_olci'].append(fpath)
        elif 'CO' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['s5p_co'].append(fpath)
        elif 'NO2' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['s5p_no2'].append(fpath)
        elif 'SO2' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['s5p_so2'].append(fpath)
        elif 'O3' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['s5p_o3'].append(fpath)
        elif 'dem' in modality_dir:
            grid_dir = os.path.join(root_dir, modality_dir, fnames_json[grid_id]['grid_id_coord'])
            if os.path.exists(grid_dir):
                fnames = os.listdir(grid_dir)
                for fname in fnames:
                    if fname.endswith('.tif'):
                        fpath = os.path.join(grid_dir, fname)
                        fpath = fpath.replace(root_dir, '')
                        fnames_json[grid_id]['dem'].append(fpath)
    count = counter.update(1)
    if count % 1000 == 0:
        print(f"Processed {count} locations in {time.time() - start_time:.3f}s.")


indices = list(fnames_json.keys())

if num_workers == 0:
    for i in indices:
        worker(i)
else:
    # parallelism data
    with Pool(processes=num_workers) as p:
        p.map(worker, indices)


# Save the json file
#with open(out_path, 'w') as f:
#    json.dump(fnames_json, f, indent=4)

# with gzip.open(out_path_all, 'wt', encoding='utf-8') as gz_file:
#     json.dump(fnames_json, gz_file)

fnames_json_union = {}
for grid_id in tqdm(fnames_json.keys()):
    if fnames_json[grid_id]['s1_grd'] or fnames_json[grid_id]['s2_toa'] or fnames_json[grid_id]['s3_olci'] or fnames_json[grid_id]['s5p_co'] or fnames_json[grid_id]['s5p_no2'] or fnames_json[grid_id]['s5p_so2'] or fnames_json[grid_id]['s5p_o3'] or fnames_json[grid_id]['dem']:
        fnames_json_union[grid_id] = fnames_json[grid_id]

with gzip.open(out_path_union, 'wt', encoding='utf-8') as gz_file:
    json.dump(fnames_json_union, gz_file)

print(f"Number of grids with at least one modality: {len(fnames_json_union)}")