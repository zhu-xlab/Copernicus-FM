#!/bin/bash

input_folder="../../../data/example_100_grids/s3_olci_utm/images/"
output_folder="../../../data/example_100_grids/s3_olci_wgs/images/"
target_crs="EPSG:4326"

mkdir -p "$output_folder"

# Function to reproject, compress, and tile a single file
reproject_file() {
    local file="$1"
    local relative_path="${file#$input_folder}"
    local output_file="$output_folder$relative_path"

    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"

    # Apply reprojection, compression, and tiling with gdalwarp
    gdalwarp -t_srs "$target_crs" -co COMPRESS=DEFLATE -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 "$file" "$output_file"

    echo "Reprojected $file -> $output_file"
}

export -f reproject_file
export input_folder output_folder target_crs

# Use GNU Parallel to process files in parallel
find "$input_folder" -type f -name "*.tif" | xargs -I {} -P 8 bash -c 'reproject_file "$@"' _ {}
