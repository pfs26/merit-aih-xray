#!/bin/bash
#SBATCH --job-name=tb_train
#SBATCH --partition=merit
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00


source ~/miniconda3/etc/profile.d/conda.sh

# conda create -n tb-env python=3.10 -y
conda activate tb-env

pip install torchinfo torchvision opencv-python matplotlib kagglehub seaborn scikit-learn

python main.py
