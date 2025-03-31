#!/bin/bash
#SBATCH --job-name=phytocnn_xcep
#SBATCH --output=logs/cnn_train_x2.out
#SBATCH --error=logs/cnn_train_x2.err
#SBATCH --time=72:00:00               # Maximum runtime of 24 hours (adjust as needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # Use 8 CPU cores per task
#SBATCH --mem=75G                     # Request 32GB of system memory
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pcdaniel@ucsc.edu
#SBATCH --partition=96x24gpu4                 # Request 1 GPU
#SBATCH --gres=gpu:p100:1


# This SLURM script is configured for training a ViT model on 1.2 million images (~18GB zipped data).
# Adjust the resources (time, CPUs, memory) if your experiments require more or less.

# Load any necessary modules (e.g., CUDA)
module load miniconda3
module load cuda/12.6              # Adjust CUDA version if needed

# Activate your conda environment
conda activate trainViT

# Run the training script with a configuration file that sets the training options.
# export CUDA_VISIBLE_DEVICES=-1

python src/train_cnn.py --config experiments/config_cnn_hb_x2.json

# Testing GPU Access
# python src/tensorflow-load-test.py

