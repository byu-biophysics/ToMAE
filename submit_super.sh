#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --mail-user=mwh1998@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs

# some helpful debugging options
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "ToMae"

python train_vanilla_auto_encoder.py

