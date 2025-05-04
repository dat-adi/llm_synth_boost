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

## Statistics Gathered

Evaluated on the `dogtooth/default_project_dev_test`, the perplexities retrieved
for various models are depicted below:
| Model | Parameters | Perplexity | GPU Memory |
| TinyLlama-1.1B | 1.1B | 10.959 | 7GB |
| Qwen2.5-3B | 3B | 10.344 | 12GB |
| Llama-3.2-3B | 3B | 10.423 | 13GB |
| Llama-3.2-3B-Instruct | 3B | 12.632 | 13GB |
| OLMoE-1B-7B | 7B | 9.71 | 22GB |
| OLMoE-1B-7B (8bit) | 7B | 9.750 | 7.1GB |
| OLMoE-1B-7B (4bit) | 7B | 10.526 | 3.7GB |
| Llama3.1 8B (8bit) | 8B | 7.983 | 9GB |
