#!/bin/bash

base_dirs=("s1_grd_utm/images" "s2_toa_mix/images" "s3_olci_wgs/images" "dem_grid_wgs/images" "s5p_l3_wgs/images/CO_column_number_density" "s5p_l3_wgs/images/O3_column_number_density" "s5p_l3_wgs/images/SO2_column_number_density" "s5p_l3_wgs/images/tropospheric_NO2_column_number_density")  # Add all sX/images directories
csv_file="grid_id_coords.csv"                             # CSV file with folder names
batch_size=10                                # Number of subfolders per tar
output_dir="raw_data_compressed"             # Output directory
processed_log="processed.log"                # Log file to track completed folders
output_prefix=""                                 # Placeholder for naming
folder_list=()                                   # Array to store folder names

# Read folder names from CSV (ignoring empty lines)
mapfile -t subfolders < <(awk 'NF' "$csv_file")

# remove the lastly generated tar.zst file if it exists
if [[ -d "$output_dir" ]]; then
    rm -f "$(ls -t "$output_dir"/*.tar.zst 2>/dev/null | head -n 1)"
else
    mkdir -p "$output_dir"
fi

# Extract processed folders from tar archives
> "$processed_log"  # Clear existing log
for file in raw_data_compressed/*.tar.zst; do
    if [[ -f "$file" ]]; then
        tar --zstd -tf "$file" | awk -F'/' '{if ($1 == "s5p_l3_wgs") print $1"/"$2"/"$3"/"$4; else print $1"/"$2"/"$3}' | sort -u >> "$processed_log"
    fi
done
sort -u -o "$processed_log" "$processed_log"  # Remove duplicates and sort

# Ensure processed log file exists
touch "$processed_log"

# Load processed folders into an array
mapfile -t processed_folders < "$processed_log"

# Create associative array for faster lookups
declare -A processed_map
for folder in "${processed_folders[@]}"; do
    processed_map["$folder"]=1
done

# Loop through each sX/images directory
for base in "${base_dirs[@]}"; do
    counter=1   # Reset tar file counter for each sX
    folder_list=()
    total_folders=${#subfolders[@]}  # Total folders for progress tracking
    processed_count=0             # Number of folders processed

    for subfolder in "${subfolders[@]}"; do
        folder_path="$base/$subfolder"
        if [[ "$base" == *"CO_column_number_density"* ]]; then
            output_prefix="s5p_co_wgs"
        elif [[ "$base" == *"O3_column_number_density"* ]]; then
            output_prefix="s5p_o3_wgs"
        elif [[ "$base" == *"SO2_column_number_density"* ]]; then
            output_prefix="s5p_so2_wgs"
        elif [[ "$base" == *"tropospheric_NO2_column_number_density"* ]]; then
            output_prefix="s5p_no2_wgs"
        else
            output_prefix="$(basename "$(dirname "$base")")"
        fi
        tar_name="${output_prefix}_$(printf "%03d" $counter).tar.zst"

        # Check if folder exists before adding
        if [[ -d "$folder_path" ]]; then
            folder_list+=("$folder_path")
        fi

        # Skip already processed folders
        if [[ -n "${processed_map[$folder_path]}" ]]; then
            if (( ${#folder_list[@]} == batch_size )); then
                echo "Skipping $tar_name (already processed)"
                folder_list=()
                ((counter++))
                processed_count=$((processed_count + batch_size))
            fi
            #echo "Skipping $folder_path (already processed)"
            continue
        fi

        # Every 'batch_size' folders, create a new tar.zst file
        if (( ${#folder_list[@]} == batch_size )); then
            tar --zstd -cf "$output_dir/$tar_name" "${folder_list[@]}"
            printf "%s\n" "${folder_list[@]}" >> "$processed_log"  # Log processed folders
            folder_list=()
            ((counter++))
            processed_count=$((processed_count + batch_size))
            echo "Processed $processed_count of $total_folders folders in $base"
        fi
    done

    # If there are remaining folders that didn't fit into a full batch
    if (( ${#folder_list[@]} > 0 )); then
        tar_name="${output_prefix}_$(printf "%03d" $counter).tar.zst"
        tar --zstd -cf "$output_dir/$tar_name" "${folder_list[@]}"
        printf "%s\n" "${folder_list[@]}" >> "$processed_log"  # Log processed folders
    fi

done

echo "Done! Tar files created."
