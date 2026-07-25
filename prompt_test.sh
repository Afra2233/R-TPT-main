#!/bin/bash
#SBATCH --job-name=rtpt_prompttest
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

# clean
# python prompt_test.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
#     --test_sets Caltech101 \
#     --gpu 0 \
#     --eps 0 \
#     --steps 0 \
#     --tta_steps 1 \
#     --diagnostics \
#     --diag_max_samples 50 \
#     --diag_grad_views 6
    
# adv
python prompt_test.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
    --test_sets UCF101 \
    --gpu 0 \
    --eps 1 \
    --steps 7 \
    --tta_steps 1 \
    --diagnostics \
    --diag_max_samples 50 \
    --diag_grad_views 6