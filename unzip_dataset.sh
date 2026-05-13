#!/bin/bash
#SBATCH --job-name=unzip_cal101
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=unzip_cal101_%j.out
#SBATCH --error=unzip_cal101_%j.err


#!/bin/bash
#SBATCH --job-name=download_ucf101
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=download_ucf101_%j.out
#SBATCH --error=download_ucf101_%j.err

module add anaconda3/2022.05
source activate rtpt


echo "Job started on $(hostname)"
echo "Start time: $(date)"

DATA_DIR="/scratch/hpc/07/zhang303/R-TPT-main/dataset/dtd"
FILE_ID="1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || {
    echo "ERROR: Cannot cd to $DATA_DIR"
    exit 1
}

echo "Current directory: $(pwd)"

# Make sure gdown is available
python -m pip install --user gdown

echo "Start downloading UCF101 from Google Drive..."
python -m gdown --id "$FILE_ID"

echo "Download finished."
echo "Files in directory:"
ls -lh

echo "End time: $(date)"

# set -e

# DATA_DIR="/scratch/hpc/07/zhang303/R-TPT-main/dataset/ucf101"

# echo "Job started on $(hostname)"
# echo "Start time: $(date)"

# cd "$DATA_DIR"

# echo "Current directory: $(pwd)"
# echo "Files before unzip:"
# ls -lh

# echo "Unzipping UCF-101-midframes.zip..."
# unzip -q UCF-101-midframes.zip

# echo "Files after unzip:"
# ls -lh

# echo "Done."
# echo "End time: $(date)"

# ======================caltech-101 =============================
# DATA_DIR="/scratch/hpc/07/zhang303/R-TPT-main/dataset/caltech-101"

# echo "Job started on $(hostname)"
# echo "Start time: $(date)"

# cd "$DATA_DIR" || {
#     echo "ERROR: Cannot cd to $DATA_DIR"
#     exit 1
# }

# echo "Current directory: $(pwd)"
# echo "Files before extraction:"
# ls -lh

# if [ -f "101_ObjectCategories.tar.gz" ]; then
#     echo "Extracting 101_ObjectCategories.tar.gz..."
#     tar -xzf 101_ObjectCategories.tar.gz
# else
#     echo "ERROR: 101_ObjectCategories.tar.gz not found in $DATA_DIR"
#     exit 1
# fi

# echo "Files after extraction:"
# ls -lh

# echo "Done."
# echo "End time: $(date)"
# ======================caltech-101 =============================