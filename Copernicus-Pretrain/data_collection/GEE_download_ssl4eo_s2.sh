python GEE_download_ssl4eo_s2.py \
--save-dir ./data/s2 \
--collection COPERNICUS/S2 \
--dates 2022-12-21 2022-09-22 2022-06-21 2022-03-20 \
--radius 1320 \
--num-workers 10 \
--log-freq 10 \
--match-file sampled_locations/sampled_locations_s2_new_100k_grid.csv \
--indices-range 1000000 1117544 \
#--resume ./data/s2/checked_locations.csv