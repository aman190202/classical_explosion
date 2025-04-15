#!/bin/bash
#SBATCH --job-name=explosion_debug
#SBATCH --output=output/debug_%j.out
#SBATCH --error=output/debug_%j.err
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1  # Debugging is typically single-threaded
#SBATCH --mem=8G
#SBATCH --partition=debug  # Use debug partition if available

echo "Debug job started at $(date)"

# Load required modules
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate graphics

# Compile with debug symbols
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..

# Create GDB command file
cat > gdb_commands.txt << EOF
set logging on
set logging file gdb_output.txt
run v1/untitled.filecache1_v1.0007.vdb
bt
info locals
quit
EOF

# Run with GDB
gdb -x gdb_commands.txt ./build/explosion

# Alternative: Run with Valgrind for memory checking
# valgrind --leak-check=full --track-origins=yes --log-file=valgrind_output.txt ./build/explosion v1/untitled.filecache1_v1.0007.vdb

echo "Debug job finished at $(date)" 