#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="NLP SSM Project"
#SBATCH --output=gemma-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate nlpssm_proj # activate the Python environment

# runs your code
python -u main.py "google/gemma-2b" >> proj.out
