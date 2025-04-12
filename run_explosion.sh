#!/bin/bash

# Build the project if needed
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake ..
    make
    cd ..
fi

# Create output directory if it doesn't exist
mkdir -p output

# Get the first 25 VDB files
vdb_files=$(ls v1/untitled.filecache1_v1.*.vdb | sort | head -n 240)

# Run explosion on each VDB file
for vdb_file in $vdb_files; do
    echo "Processing $vdb_file..."
    ./build/explosion "$vdb_file"
done

echo "Done processing all VDB files!" 