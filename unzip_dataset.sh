#!/bin/bash
#SBATCH --job-name=unzip_cal101
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=unzip_cal101_%j.out
#SBATCH --error=unzip_cal101_%j.err

DATA_DIR="/scratch/hpc/07/zhang303/R-TPT-main/dataset/caltech-101"

echo "Job started on $(hostname)"
echo "Start time: $(date)"

cd "$DATA_DIR" || {
    echo "ERROR: Cannot cd to $DATA_DIR"
    exit 1
}

echo "Current directory: $(pwd)"
echo "Files before extraction:"
ls -lh

if [ -f "101_ObjectCategories.tar.gz" ]; then
    echo "Extracting 101_ObjectCategories.tar.gz..."
    tar -xzf 101_ObjectCategories.tar.gz
else
    echo "ERROR: 101_ObjectCategories.tar.gz not found in $DATA_DIR"
    exit 1
fi

echo "Files after extraction:"
ls -lh

echo "Done."
echo "End time: $(date)"