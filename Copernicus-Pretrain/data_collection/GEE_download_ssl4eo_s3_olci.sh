python GEE_download_ssl4eo_s3_olci.py \
--save-dir ./data/s3_olci \
--collection COPERNICUS/S3/OLCI \
--cloud-pct 20 \
--dates 2021-01-01 2021-12-31 \
--radius 14000 \
--sequence-length 8 \
--num-workers 20 \
--log-freq 10 \
--match-file sampled_locations/grid_index_land_s3.csv \
--indices-range 127170 980320 \
#--resume ./data/s3_olci/checked_locations.csv