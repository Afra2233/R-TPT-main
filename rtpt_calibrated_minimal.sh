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

python rtpt_disagreement_entropy_both.py \
  /scratch/hpc/07/zhang303/R-TPT-main/dataset \
  --test_sets UCF101 \
  -a RN50 \
  -b 64 \
  --gpu 0 \
  --ctx_init a_photo_of_a \
  -p 50 \
  --eps 1.0 \
  --steps 7 \
  --tta_steps 1 \
  --eval_both \
  --selection_p 0.1 \
  --agreement_strength 1.0 \
  --agreement_temp 0.20 \
  --entropy_floor_base 0.02 \
  --entropy_floor_scale 0.08 \
  --sample_lr_min 0.25 \
  --sample_lr_power 1.0 \
  --density_temp 0.01 \
  --num_neighbors 20 \
  --ece_bins 15 \
  --output_dir output_results/rtpt_js_entropy_both