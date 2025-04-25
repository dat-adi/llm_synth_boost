## Installation Instructions

The following are set up instructions.

```python
# 0. Load anaconda into your environment (for server usage)
module load anaconda

# 1. Create a conda environment
conda create -n moepi python=3.10

# 2. Activate environment
conda activate moepi

# 3. Install dependencies
pip install -r requirements.txt
```

## Execution Instructions

Ensure that your environment is active.
```python
# 0. Environment activation
module load anaconda
conda activate moepi

# 1. (Optional) Create sbatch for required model
cp gemma.sh new_model.sh
# Edit as per requirement (both the file name and model name fed to the script)

# 2. Run the model on the GPU using Slurm
sbatch new_model.sh
```

The error/debug output is pushed towards `new_model-%j.out` and the print output
is pushed towards `proj.out`. This is not an architecture choice, just for
convenience. Feel free to change it.
