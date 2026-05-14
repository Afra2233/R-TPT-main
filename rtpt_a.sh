#!/bin/bash
#SBATCH --job-name=rtpt_clean
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

# python rtpt_a.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
#   --test_sets Caltech101 \
#   -a RN50 \
#   -b 64 \
#   --gpu 0 \
#   --ctx_init a_photo_of_a \
#   -p 1000 \
#   --eps 0.0 \
#   --output_dir output_results/dir_cons_entropy_clean \
#   --ece_bins 15 \
#   --high_conf_th 0.9 \
#   --dir_conservative_entropy \
#   --dir_temp 1.0 \
#   --alpha_offset 1.0 \
#   --dir_gate_tau 0.1 \
#   --lambda_cons 1.0



python rtpt_a.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
  --test_sets Caltech101 \
  -a RN50 \
  -b 64 \
  --gpu 0 \
  --ctx_init a_photo_of_a \
  -p 1000 \
  --eps 1.0 \
  --steps 7 \
  --output_dir output_results/dir_cons_entropy_robust \
  --ece_bins 15 \
  --high_conf_th 0.9 \
  --dir_conservative_entropy \
  --dir_temp 1.0 \
  --alpha_offset 1.0 \
  --dir_gate_tau 0.1 \
  --lambda_cons 1.0


  