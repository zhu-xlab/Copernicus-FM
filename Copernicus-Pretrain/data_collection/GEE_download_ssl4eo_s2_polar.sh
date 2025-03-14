python GEE_download_ssl4eo_s2.py \
--save-dir ./data/s2_polar \
--collection COPERNICUS/S2_HARMONIZED \
--dates 2023-03-20 2022-03-20 2021-03-20 2020-03-20 \
--radius 1320 \
--num-workers 10 \
--log-freq 10 \
--match-file sampled_locations/sampled_locations_s2_missing_grids_diff_polar.csv \
--indices-range 1200001 1240085 \
#--resume ./data/s2/checked_locations.csv