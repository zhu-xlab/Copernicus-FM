python GEE_download_ssl4eo_s1.py \
--save-dir ./data/s1 \
--collection COPERNICUS/S1_GRD \
--dates 2022-12-21 2022-09-22 2022-06-21 2022-03-20 \
--radius 1320 \
--num-workers 10 \
--log-freq 10 \
--match-file sampled_locations/sampled_locations_s2_plus_downloaded_741k_grid.csv \
--indices-range 131 1999998 \
#--resume ./data/s5p/checked_locations.csv