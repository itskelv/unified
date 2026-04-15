#!/bin/bash
#SBATCH --job-name=unidcase
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err

module load cuda/12.1
conda activate uni
python3 batch_feature_extraction.py 3
python3 train_seldnet.py 3