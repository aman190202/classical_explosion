#!/bin/bash
#SBATCH --job-name=explosion_render_light
#SBATCH --output=output/render_l%j.out
#SBATCH --error=output/render_k%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G





echo "Job started at $(date)"

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate graphics
cd build && cmake .. && make && cd ..
./build/explosion v1/untitled.filecache1_v1.0024.vdb

# Print completion time
echo "Job finished at $(date)" 