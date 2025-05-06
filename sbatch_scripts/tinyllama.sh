#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="NLP SSM Project"
#SBATCH --output=tinyllama-%j.out
#SBATCH --mem=16G

module load anaconda
conda activate nlpssm_proj # activate the Python environment

# runs your code
python -u main.py "TinyLlama/TinyLlama-1.1B-step-50K-105b" >> proj.out
