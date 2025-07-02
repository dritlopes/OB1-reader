#!/usr/bin/env bash

# Set job requirements
#SBATCH -J train_classifier
#SBATCH -N 1
#SBATCH -p defq
#SBATCH --gpus=1
#SBATCH -w node011
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.t.lopesrego@vu.nl

# Environment modules
module load shared 2024
module load 2024 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python $HOME/SaliencyMap/train_classifier.py
