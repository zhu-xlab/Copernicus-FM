# Download Copernicus-Pretrain dataset

This directory contains codes (to be cleaned!) to collect the Copernicus-Pretrain dataset. Generally, we structure the dataset acoording to [ERA5 0.25x0.25 grids](https://confluence.ecmwf.int/display/CKB/ERA5%253A+What+is+the+spatial+reference) (721x1440, about 1M in total, about 300k in land), and for each grid download corresponding Sentinel-1/2/3/5P/DEM data.

## Sample grids and patch locations

We first sample regional grids as downloading units, c.f. `utils/sample_era5_grids_land.ipynb`. To start, we sample all ERA5 grids (about 1M), and filter land grids with coaster buffer (about 390K). We use S3 as the anchor to connect all modalities, further filterring the lat range to [85N, 74S] (about 310K, mainly reducing interior Antarctica where almost no satellite observation is available). The final grids cover global land and near-land ocean, including polar regions.

For S3, S5P, and DEM, download according to the entire grids. For S1/S2, we don't cover the whole grid area but instead further sample local patches within the grids. To increase the landscape diversity, we use Gaussian sampling to sample 1M locations (including SSL4EO-S12 251K, c.f. the sampling script of [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) or [SSL4EO-L](https://github.com/microsoft/torchgeo/tree/main/experiments/ssl4eo)), download S1/S2 local patches for all available locations, and organize them into ERA5 grids. After that, we uniform sample around center locations of the still empty grids to complete.

You will need to get the following files (exact filenames may differ) to start downloading:

- `grid_index_land_s3.csv` for S3/5P/DEM
- `sampled_locations_s2.csv` for S1/S2

## Download images with GEE

We use Google Earth Engine to download the images.

- To download S2 (264x264x13x4), run `GEE_download_ssl4eo_s2.sh` which downloads a sequence for each local patch location and automatically organizes them into the 0.25x0.25 grids. Default S1 CRS is UTM.
- To download S1 (264x264x2x4), run `GEE_download_ssl4eo_s1.sh` which matches the S2 locations for downloading. Default S2 CRS is UTM.
- To download S3 (96x96x21x8), run `GEE_download_ssl4eo_s3.sh` which downloads a sequence for each land grid. Default S3 CRS is UTM, but we preferrably use WGS84, allowing distortion in high latitudes but better aligning ERA5 grids.
- To download S5P (28x28x4x12), run `GEE_download_ssl4eo_s5p.sh` which downloads a sequence for each land grid for each variable of [CO, NO2, O3, SO2]. Each image is monthly mean. Default S5P CRS is WGS84.
- To download DEM (960x960), run `GEE_download_ssl4eo_dem.sh` which downloads a single patch for each land grid. Default DEM CRS is WGS84.
- We additionally download S1/2 images for polar regions, run `GEE_download_ssl4eo_s1_polar.sh` (EW mode) and `GEE_download_ssl4eo_s2_polar.sh`.

Data structure is as follows:

```bash
data
├── s1
│   ├── images
│   │   ├── gridId_lon_lat
│   │   │   ├── S2Id_lon_lat
│   │   │   │   ├── GEEID_1.tif
│   │   │   │   ├── GEEID_2.tif
│   │   │   │   ├── GEEID_3.tif
│   │   │   │   ├── GEEID_4.tif
│   │   │   ├── ...
│   |   ├── ...
│   ├── checked_locations.csv
├── s2
|   ├── ...
├── s3
|   ├── images
│   │   ├── gridId_lon_lat
│   │   │   ├── GEEID_1.tif
│   │   │   ├── ...
│   |   ├── ...
│   ├── checked_locations.csv
├── s5p
|   ├── images
│   │   ├── CO_column_number_density
│   │   │   ├── gridId_lon_lat
│   │   │   │   ├── 2021-M1-01_2021-M2-01.tif
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── NO2_column_number_density
│   │   ├── O3_column_number_density
│   │   ├── SO2_column_number_density
│   ├── checked_locations.csv
├── dem
|   ├── images
│   │   ├── gridId_lon_lat
│   │   │   ├── dem.tif
│   │   ├── ...
│   ├── checked_locations.csv
```

For the first time using Google Earth Engine, you need to authenticate your account. After `pip install earthengine-api`, you can either

- run `earthengine authenticate` and follow the instructions (add `--auth_mode notebook` when having trouble opening the browser in a remote server);
- or start a python interface and run `import ee` and `ee.Authenticate()`, then follow the instructions.

Some tips for downloading:

- To split all the grids to several runs, take a subset of the full indices by modifying the argument `--indices-range`.
- **To resume from interruption, add `--resume ./data/checked_locations.csv`.**

## Quality check and filtering

We can organize all file paths in a json file for fast access later. Run `utils/count_all_fnames.py` to generate `fnames_all.json.gz`, which is a dictionary with all file paths falling into the full ~1M ERA5 grids (most ocean grids will be empty). We can simplify the json file later.

Some rounds of check are needed to ensure the downloaded data is correct and complete, including:

- some files can be corrupted and need to be removed (large quantity for S5P);
- "most" S3 grids have 8 timestamps, "most" S1/S2 local patches have 4 timestamps;
- "most" S1/S2 patches are paired;

Run `utils/remove_corrupt.py` to remove corrupted files. Run `utils/count_all_fnames_filter.py` to do the filtering (not must, only when downloaded S1/2/3 images go too extreme) and generate `sampled_locations/fnames_all_new.json.gz`. Additionally, we can generate `fnames_all_union.json.gz` to extract grids with at least one modality available, and generate `fnames_all_aligned.json.gz` to extract grids with all modalities available.

## Visualize and analyze

Run `utils/count_all_fnames_vis.ipynb` to visualize the dataset distribution, and analyze some statistics.
