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
# clean:
python rtpt_ece.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
  --test_sets DTD \
  -a RN50 \
  -b 64 \
  --gpu 0 \
  --ctx_init a_photo_of_a \
  -p 50 \
  --eps 0.0 \
  --output_dir output_results/sc_rtpt_clean \
  --sc_view_filter \
  --sc_anomaly_ratio 0.1 \
  --sc_anomaly_penalty 0.05 \
  --rtpt_tau 0.01

# adv
# python rtpt_ece.py /scratch/hpc/07/zhang303/R-TPT-main/dataset --test_sets eurosat  -a RN50 -b 64 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 1.0 --step 7 --output_dir 'output_results/rtpt'
# python rtpt_ece.py /scratch/hpc/07/zhang303/R-TPT-main/dataset \
#   --test_sets DTD \
#   -a RN50 \
#   -b 64 \
#   --gpu 0 \
#   --ctx_init a_photo_of_a \
#   -p 50 \
#   --eps 1.0 \
#   --steps 7 \
#   --output_dir output_results/sc_rtpt_robust \
#   --sc_view_filter \
#   --sc_anomaly_ratio 0.1 \
#   --sc_anomaly_penalty 0.05 \
#   --rtpt_tau 0.01

# fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 
#                     'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']


