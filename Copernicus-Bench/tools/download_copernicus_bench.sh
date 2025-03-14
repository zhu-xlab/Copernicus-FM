#!/bin/bash

BASE_URL="https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main"


# List of file URLs
files=(
    "l1_cloud_s2/cloud_s2.zip"
    "l1_cloud_s3/cloud_s3.zip"
    "l2_eurosat_s1s2/eurosat_s1.zip"
    "l2_eurosat_s1s2/eurosat_s2.zip"
    "l2_bigearthnet_s1s2/bigearthnetv2-s1-10%25.tar.zst"
    "l2_bigearthnet_s1s2/bigearthnetv2-s2-10%25.tar.zst"
    "l2_bigearthnet_s1s2/metadata-5%25.parquet"
    "l2_dfc2020_s1s2/dfc2020.zip"
    "l2_lc100_s3/lc100_s3.zip"
    "l3_flood_s1/flood_s1.zip"
    "l3_lcz_s2/lcz_train.h5"
    "l3_lcz_s2/lcz_val.h5"
    "l3_lcz_s2/lcz_test.h5"
    "l3_biomass_s3/biomass_s3.zip"
    "l3_airquality_s5p/airquality_s5p.zip"
)



# Destination directory (current directory)
dest_dir="./data/copernicusbench"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through each URL
for file_path in "${files[@]}"; do

    url="$BASE_URL/$file_path"  # Construct full URL
    filename=$(basename "$file_path")

    echo "Downloading: $filename ..."
    wget -q --show-progress -O "$dest_dir/$filename" "$url"

    # If the file is a zip, unzip it
    if [[ "$filename" == *.zip ]]; then
        echo "Extracting: $filename ..."
        unzip -o "$dest_dir/$filename" -d "$dest_dir"
        rm "$dest_dir/$filename"  # Delete the ZIP file after extraction
    fi

    # If the file is a tar.zst, extract it to "bigearthnet_s1s2" directory
    if [[ "$filename" == *.tar.zst ]]; then
        echo "Extracting: $filename ..."
        mkdir -p "$dest_dir/bigearthnet_s1s2"
        tar -I zstd -xf "$dest_dir/$filename" -C "$dest_dir/bigearthnet_s1s2"
        rm "$dest_dir/$filename"  # Delete the TAR.ZST file after extraction
    fi

    # If the file is a parquet, move it to "bigearthnet_s1s2" directory
    if [[ "$filename" == *.parquet ]]; then
        echo "Moving: $filename ..."
        mv "$dest_dir/$filename" "$dest_dir/bigearthnet_s1s2"
    fi

    # If the file is an HDF5, move it to "lcz_s2" directory
    if [[ "$filename" == *.h5 ]]; then
        echo "Moving: $filename ..."
        mkdir -p "$dest_dir/lcz_s2"
        mv "$dest_dir/$filename" "$dest_dir/lcz_s2"
    fi

done

echo "✅ All files downloaded and extracted!"
