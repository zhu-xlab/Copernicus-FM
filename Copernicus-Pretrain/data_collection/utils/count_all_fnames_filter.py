import os
import json
import csv
from tqdm import tqdm
import gzip
import shutil
import copy

fnames_all_fpath = '../../data/example_100_grids_cleaned/fnames_sampled_all.json.gz'
root_dir = '../../data/example_100_grids_cleaned/'
new_fnames_all_fpath = '../../data/example_100_grids_cleaned/fnames_sampled_all_new.json.gz'
fnames_all_cleaned_fpath = '../../data/example_100_grids_cleaned/fnames_sampled_cleaned.json.gz'

### read the compressed json file
with gzip.open(fnames_all_fpath, 'rt', encoding='utf-8') as gz_file:
    fnames_json = json.load(gz_file)
# create a new copy of the json file
fnames_json_copy = copy.deepcopy(fnames_json)

# ### filter out those s3 grids that do not have 8 files
# count_s3 = 0
# for grid_id in tqdm(fnames_json.keys()):
#     s3_grid_dir = os.path.join(root_dir, 's3_olci_utm/images', fnames_json[grid_id]['grid_id_coord'])
#     file_paths = fnames_json[grid_id]['s3_olci']
#     num_files = len(file_paths)

#     if num_files == 0:
#         continue
#     elif num_files < 8:
#         fnames_json_copy[grid_id]['s3_olci'] = []
#         # also delete the files in the system
#         shutil.rmtree(s3_grid_dir)
#         print(f"Deleted S3 grid {grid_id} as lack of enough files")
#         count_s3 += 1
#     elif num_files > 8:
#         remaining_files = file_paths[:8]
#         fnames_json_copy[grid_id]['s3_olci'] = remaining_files
#         # also delete the extra files in the system
#         for file_path in file_paths[8:]:
#             if os.path.exists(os.path.join(root_dir,file_path)):
#                 os.remove(os.path.join(root_dir,file_path))
#         print(f"Deleted extra S3 grid {grid_id} files")
#         count_s3 += 1
# print(f"Number of S3 grids modified: {count_s3}")

# ### filter out those s1/s2 local grids that do not match each other
# count_s1 = 0
# count_s2 = 0
# for grid_id in tqdm(fnames_json.keys()):
#     s2_grid_dir = os.path.join(root_dir, 's2_toa_mix/images', fnames_json[grid_id]['grid_id_coord'])
#     s1_grid_dir = os.path.join(root_dir, 's1_grd_utm/images', fnames_json[grid_id]['grid_id_coord'])
#     for s1_id in fnames_json[grid_id]['s1_grd'].keys():
#         if s1_id not in fnames_json[grid_id]['s2_toa'].keys():
#             print(f"Grid {grid_id} has missing s2 local id {s1_id}")
#             count_s1 += 1
#             # remove the s1 id from the json
#             del fnames_json_copy[grid_id]['s1_grd'][s1_id]
#             # also delete the file in the system
#             local_dir = os.path.join(s1_grid_dir, s1_id)
#             shutil.rmtree(local_dir)
#     for s2_id in fnames_json[grid_id]['s2_toa'].keys():
#         if s2_id not in fnames_json[grid_id]['s1_grd'].keys():
#             print(f"Grid {grid_id} has missing s1 local id {s2_id}")
#             count_s2 += 1
#             # remove the s2 id from the json
#             del fnames_json_copy[grid_id]['s2_toa'][s2_id]
#             # also delete the file in the system
#             local_dir = os.path.join(s2_grid_dir, s2_id)
#             shutil.rmtree(local_dir)
# print(f"Number of s1 grids modified: {count_s1}")
# print(f"Number of s2 grids modified: {count_s2}")


# save new json file
with gzip.open(new_fnames_all_fpath, 'wt', encoding='utf-8') as gz_file:
    json.dump(fnames_json_copy, gz_file)

# remove entries with at least one empty modality
fnames_json_cleaned = {}
for grid_id in tqdm(fnames_json_copy.keys()):
    if fnames_json_copy[grid_id]['s1_grd'] and fnames_json_copy[grid_id]['s2_toa'] and fnames_json_copy[grid_id]['s3_olci'] and fnames_json_copy[grid_id]['s5p_co'] and fnames_json_copy[grid_id]['s5p_no2'] and fnames_json_copy[grid_id]['s5p_so2'] and fnames_json_copy[grid_id]['s5p_o3'] and fnames_json_copy[grid_id]['dem']:
        fnames_json_cleaned[grid_id] = fnames_json_copy[grid_id]

print(f"Number of grids with all modalities: {len(fnames_json_cleaned)}")

### save the cleaned json file
with gzip.open(fnames_all_cleaned_fpath, 'wt', encoding='utf-8') as gz_file:
    json.dump(fnames_json_cleaned, gz_file)