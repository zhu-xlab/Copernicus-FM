python GEE_download_ssl4eo_dem.py \
--save-dir ./data/dem_grid \
--collection COPERNICUS/DEM/GLO30 \
--radius 14000 \
--num-workers 20 \
--log-freq 10 \
--match-file sampled_locations/grid_index_land_s3.csv \
--indices-range 127170 980320 \
#--resume ./data/dem_grid/checked_locations.csv