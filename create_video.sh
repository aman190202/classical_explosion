#!/bin/bash

# Check if output directory exists and has PPM files
if [ ! -d "output" ] || [ -z "$(ls output/*.ppm 2>/dev/null)" ]; then
    echo "No PPM files found in output directory. Please run run_explosion.sh first."
    exit 1
fi

# Create a temporary directory for numbered frames
mkdir -p temp_frames

# Copy and rename PPM files with padded numbers
counter=0
for file in $(ls output/*.ppm | sort -V); do
    new_name=$(printf "temp_frames/frame_%04d.ppm" $counter)
    cp "$file" "$new_name"
    ((counter++))
done

# Use ffmpeg to create the video
echo "Creating video from frames..."
ffmpeg -framerate 24 -i temp_frames/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p -y explosion.mp4

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf temp_frames

echo "Video created: explosion.mp4" 