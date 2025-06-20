#!/bin/bash

# Script to process depth map PNG files
# Goes through each subdirectory, finds PNGs, renames them systematically, and zips the result

SOURCE_DIR="bucket/analysis-pipeline-data/National_Highways/Wheatley_Tunnel/data_intermediate/figures/dataset_depth_image/20250617"
OUTPUT_DIR="depth_maps_processed"
ZIP_NAME="~/patchcore-inspection/depth_maps_collection.zip"

# # Create output directory
# if [ -d "$OUTPUT_DIR" ]; then
#     echo "Removing existing output directory..."
#     rm -rf "$OUTPUT_DIR"
# fi

# mkdir -p "$OUTPUT_DIR"
# echo "Created output directory: $OUTPUT_DIR"

# # Process each subdirectory
# for subdir in "$SOURCE_DIR"/merged_scan_*; do
#     if [ -d "$subdir" ]; then
#         subdir_name=$(basename "$subdir")
#         echo "Processing $subdir_name..."

#         # Extract scan number from directory name (e.g., merged_scan_2 -> 2)
#         scan_num=$(echo "$subdir_name" | sed 's/merged_scan_//')

#         # Create array of PNG files and sort them
#         png_files=()
#         while IFS= read -r -d '' file; do
#             png_files+=("$file")
#         done < <(find "$subdir" -name "*.png" -type f -print0 | sort -zV)

#         # Process each PNG file
#         for png_file in "${png_files[@]}"; do
#             # Extract the base filename without extension
#             base_name=$(basename "$png_file" .png)

#             # Extract the image number from the original filename (e.g., 169_perspective -> 169)
#             image_num=$(echo "$base_name" | sed 's/_perspective$//')

#             # Create new filename preserving the original image number
#             new_filename=$(printf "merged_scan_%s_%s_perspective.png" "$scan_num" "$image_num")

#             # Copy file to output directory with new name
#             cp "$png_file" "$OUTPUT_DIR/$new_filename"
#             echo "Copied: $base_name -> $new_filename"
#         done

#         echo "Completed processing $subdir_name (${#png_files[@]} files)"
#     fi
# done

# # Count total files processed
# total_files=$(find "$OUTPUT_DIR" -name "*.png" -type f | wc -l)
# echo "Total PNG files processed: $total_files"

# Create zip archive
echo "Creating zip archive: $ZIP_NAME"
cd "$OUTPUT_DIR"
zip -r "$ZIP_NAME" *.png > /dev/null 2>&1
cd ..

echo "ZIP archive created: $ZIP_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Script completed successfully!"