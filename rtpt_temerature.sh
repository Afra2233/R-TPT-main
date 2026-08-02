#!/bin/bash
#SBATCH --job-name=rtpt_adaptive
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

python rtpt_temperature.py \
  /scratch/hpc/07/zhang303/R-TPT-main/dataset \
  --test_sets Caltech101 \
  -a RN50 \
  -b 64 \
  --gpu 0 \
  --ctx_init a_photo_of_a \
  -p 50 \
  --eps 1.0 \
  --tta_steps 1 \
  --steps 7 \
  --rtpt_tau 0.01 \
  --output_dir output_results/adaptive_rtpt_temperature