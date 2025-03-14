python GEE_download_ssl4eo_s5p.py \
--save-dir ./data/s5p \
--collection COPERNICUS/S5P/OFFL/L3_CO COPERNICUS/S5P/OFFL/L3_NO2 COPERNICUS/S5P/OFFL/L3_O3 COPERNICUS/S5P/OFFL/L3_SO2 \
--band-name CO_column_number_density tropospheric_NO2_column_number_density O3_column_number_density SO2_column_number_density \
--dates 2021-01-01 2021-02-01 2021-03-01 2021-04-01 2021-05-01 2021-06-01 2021-07-01 2021-08-01 2021-09-01 2021-10-01 2021-11-01 2021-12-01 2021-12-31 \
--radius 14000 \
--sequence-length 12 \
--num-workers 40 \
--log-freq 10 \
--match-file sampled_locations/grid_index_land_s3.csv \
--indices-range 127170 980320 \
#--resume ./data/s5p/checked_locations.csv