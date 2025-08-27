#!/bin/bash

# Script to unzip all .zip files in ./data/imagenet/train/ directory
# Author: Auto-generated script
# Usage: ./unzip_imagenet_train.sh

set -e  # Exit on any error

mkdir data/
huggingface-cli download QingyuShi/ImageNet1K --local-dir ./data/imagenet/ --repo-type dataset

unzip -p ./data/imagenet/val.zip -d ./data/imagenet/

# Define the target directory
TARGET_DIR="./data/imagenet/train/"

# Find all .zip files in the target directory
ZIP_FILES=$(find "$TARGET_DIR" -name "*.zip" -type f)

# Count the number of .zip files
ZIP_COUNT=$(echo "$ZIP_FILES" | wc -l)
echo "Found $ZIP_COUNT .zip file(s) to extract"

# Process each .zip file
while IFS= read -r zip_file; do
    echo "Processing: $(basename "$zip_file")"
    
    # Get the directory where the zip file is located
    ZIP_DIR=$(dirname "$zip_file")
    
    # Get filename without extension for subdirectory name
    ZIP_NAME=$(basename "$zip_file" .zip)
    
    # Extract the zip file to its own subdirectory
    if unzip -q "$zip_file" -d "$TARGET_DIR"; then
        echo "✓ Successfully extracted: $(basename "$zip_file") to $ZIP_NAME/"
        
        # Remove the zip file after successful extraction
        rm "$zip_file"
        echo "  → Removed original zip file"
    else
        echo "✗ Failed to extract: $(basename "$zip_file")"
    fi
    echo ""
done <<< "$ZIP_FILES"

echo "All extractions completed successfully!"