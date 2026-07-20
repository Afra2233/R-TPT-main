#!/bin/bash
#SBATCH --job-name=crtpt_adv_eps1
#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/hpc/07/zhang303/R-TPT-main/%x-%j.out
#SBATCH -e /scratch/hpc/07/zhang303/R-TPT-main/%x-%j.err

module add anaconda3/2022.05
source activate rtpt

cd /scratch/hpc/07/zhang303/R-TPT-main

python rtpt_shared_random_sensitivity_ece.py \
    /path/to/dataset \
    --test_sets Caltech101 \
    --arch RN50 \
    --batch-size 64 \
    --selection_p 0.1 \
    --sensitivity_drop_ratio 0.2 \
    --sensitivity_rho 1e-3 \
    --eps 1.0 \
    --steps 7 \
    --gpu 0