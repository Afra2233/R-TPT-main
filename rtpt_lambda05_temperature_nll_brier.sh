#!/bin/bash
#SBATCH --job-name=rtpt_lambda05_temperature
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
	# DTD	Caltech101	UCF101	Flower102	Pets	Aircraft	Food101	Pets	Aircraft	eurosat	Cars	SUN397
# adv
# python rtpt_lambda05_temperature_nll_brier.py \
# /scratch/hpc/07/zhang303/R-TPT-main/dataset \
# --test_sets Aircraft \
# -a RN50 \
# -b 64 \
# --gpu 0 \
# --ctx_init a_photo_of_a \
# -p 50 \
# --eps 1.0 \
# --steps 7 \
# --output_dir 'output_results/rtpt_lambda05_nll_brier'

# clean
python rtpt_lambda05_temperature_nll_brier.py \
/scratch/hpc/07/zhang303/R-TPT-main/dataset \
--test_sets Caltech101 \
-a RN50 \
-b 64 \
--gpu 0 \
--ctx_init a_photo_of_a \
-p 50 \
--eps 0.0 \
--output_dir 'output_results/rtpt_lambda05_nll_brier_clean'