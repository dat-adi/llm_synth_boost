import json
import math
from rich import print
import numpy as np

def find_significant_perplexities(perplexities):
    # Calculate mean and standard deviation
    mean = np.mean(perplexities)
    std_dev = np.std(perplexities)
    
    # Find perplexities above 1st and 2nd standard deviation
    first_std_dev_threshold = mean + std_dev
    second_std_dev_threshold = mean + 2 * std_dev
    
    # Get outliers with their indices
    first_std_dev_outliers = [(idx, val) for idx, val in enumerate(perplexities) 
                             if val >= first_std_dev_threshold and val < second_std_dev_threshold]
    
    second_std_dev_outliers = [(idx, val) for idx, val in enumerate(perplexities) 
                              if val >= second_std_dev_threshold]
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'first_std_dev_outliers': first_std_dev_outliers,
        'second_std_dev_outliers': second_std_dev_outliers
    }

def get_largest_outliers(perplexities):
    # Calculate mean and standard deviation
    mean = sum(perplexities) / len(perplexities)
    std_dev = (sum((x - mean) ** 2 for x in perplexities) / len(perplexities)) ** 0.5
    
    # Find perplexities above 2nd standard deviation (largest outliers)
    second_std_dev_threshold = mean + 2 * std_dev
    
    # Get set of indices for the largest outliers
    largest_outliers = {idx for idx, val in enumerate(perplexities) 
                       if val >= second_std_dev_threshold}
    
    return largest_outliers

with open("./allenai-OLMoE-1B-7B-0924.ppl.out") as f:
    contents = json.load(f)

sentences = list(contents.keys())
losses = contents.values()
ppl = lambda x: math.exp(x)

ppls = [ppl(x) for x in losses]
max_index = ppls.index(max(ppls))
print(sentences[max_index], ppls[max_index])
# print("Largest ppl: ", max(ppls))
print(find_significant_perplexities(ppls))

for i in get_largest_outliers(ppls):
    print(i, sentences[i][:30] + "..." + sentences[i][-30:], ppls[i])


## Observations
"""
TinyLlama-TinyLlama-1 might not be good at code, considering that that's the only thing it's suffering the most at.
"""
