To run this code, please ensure that you have the synthetic data downloaded onto
the folder `data/` at the root level. This will be utilized for finetuning and
may be used for generating the training set based on the sampling approach 
(random, mixed, synthetic).

# Baseline Evaluation

Run the `baseline_eval.py` file with the model name provided as the argument to
receive the model's perplexity as a metric.
```bash
python baseline_eval.py [MODEL_NAME] # boilerplate
python baseline_eval.py "meta-llama/Llama-3.1-8B" # example
```

# Finetuning the Model

This file is specific and finetunes the **TinyLlama** model using **QLoRA with 8-bit
quantization and at rank 8**. The hyperparameters for this file are hard-coded and
you will need to edit it to alter parameters/change configuration. You can also
comment and uncomment the lines 100 to 121 based on sampling strategy.

Again, it is expected that you have the `data/synthetic_all.csv` file present 
in the repository for mixed and synthetic sampling.
```bash
python finetune.py
```

# Baseline vs Finetuned Evaluation

We have a single script that evaluates the baseline and the finetuned model for 
perplexity at a sentence wise level. It can be run as follows:
```bash
python eval_baseline_and_final.py
```
