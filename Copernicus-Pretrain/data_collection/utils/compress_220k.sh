#!/bin/bash

base_dirs=("s1_grd_utm/images" "s2_toa_mix/images" "s3_olci_wgs/images" "dem_grid_wgs/images" "s5p_l3_wgs/images/CO_column_number_density" "s5p_l3_wgs/images/O3_column_number_density" "s5p_l3_wgs/images/SO2_column_number_density" "s5p_l3_wgs/images/tropospheric_NO2_column_number_density")
csv_file="grid_id_coords_220k_joint.csv"
batch_size=1000                             # Number of grid IDs per tar chunk
output_dir="compressed_tar_chunks"
processed_log="processed_220k.log"

# Read folder names from CSV (ignoring empty lines)
mapfile -t grid_ids < <(awk 'NF' "$csv_file")
total_grids=${#grid_ids[@]}

# Setup directories and processed logs
if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
fi

# Extract processed folders from tar archives
> "$processed_log"  # Clear existing log
for file in raw_data_compressed/*.tar.zst; do
    if [[ -f "$file" ]]; then
        tar --zstd -tf "$file" | awk -F'/' '{if ($1 == "s5p_l3_wgs") print $1"/"$2"/"$3"/"$4; else print $1"/"$2"/"$3}' | sort -u >> "$processed_log"
    fi
done
sort -u -o "$processed_log" "$processed_log"

# Load processed folders into an array
mapfile -t processed_folders < "$processed_log"

# Create associative array for faster lookups
declare -A processed_map
for folder in "${processed_folders[@]}"; do
    processed_map["$folder"]=1
done

# Process grids in batches
total_batches=$(( (total_grids + batch_size - 1) / batch_size ))

for ((batch=0; batch<total_batches; batch++)); do
    start_idx=$((batch * batch_size))
    end_idx=$(( start_idx + batch_size - 1 ))
    
    # Ensure we don't go beyond array bounds
    if ((end_idx >= total_grids)); then
        end_idx=$((total_grids - 1))
    fi
    
    batch_num=$((batch + 1))
    
    echo "Processing batch $batch_num of $total_batches (grids ${start_idx}-${end_idx})"
    
    # Process each base directory for this batch of grid IDs
    for base in "${base_dirs[@]}"; do
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
        
        tar_name="${output_prefix}_batch$(printf "%03d" $batch_num).tar.zst"
        folder_list=()
        
        # Skip this batch if already processed
        existing_batch_file="$output_dir/$tar_name"
        if [[ -f "$existing_batch_file" ]]; then
            echo "Skipping batch $batch_num for $output_prefix (already exists)"
            continue
        fi
        
        # Process each grid ID in this batch
        for ((i=start_idx; i<=end_idx; i++)); do
            subfolder="${grid_ids[i]}"
            folder_path="$base/$subfolder"
            
            # Check if folder exists and hasn't been processed
            if [[ -d "$folder_path" && -z "${processed_map[$folder_path]}" ]]; then
                folder_list+=("$folder_path")
            fi
        done
        
        # Create tar file if we have any folders
        if (( ${#folder_list[@]} > 0 )); then
            echo "Creating $tar_name with ${#folder_list[@]} folders"
            tar --zstd -cf "$output_dir/$tar_name" "${folder_list[@]}"
            printf "%s\n" "${folder_list[@]}" >> "$processed_log"  # Log processed folders
            
            # Update processed map
            for folder in "${folder_list[@]}"; do
                processed_map["$folder"]=1
            done
        else
            echo "No folders to process for $output_prefix in batch $batch_num"
        fi
    done
done

echo "Done! Tar files created."