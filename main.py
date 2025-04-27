from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from datasets import load_dataset
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_ppl(model, tokenizer, dataset):
    """Evaluates perplexity for a model given the dataset"""

    total_loss = 0.0
    total_tokens = 0
    per_item_ppl = []
    i = 0
    for example in dataset:
        try:
            text = example["text"]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs.input_ids.to(device)

            with torch.no_grad():
              outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            per_item_ppl.append(loss.item())

            num_tokens = input_ids.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        except Exception as err:
            print(f"Skipping {i}: {err}")

        finally:
            i += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, per_item_ppl


def main():
    parser = argparse.ArgumentParser(description="Script that takes a model name")
    parser.add_argument("MODEL_NAME", type=str, help="Name of the model to use")

    args = parser.parse_args()
    MODEL_NAME = args.MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model.to(device)

    DATASET_NAME = "dogtooth/default_project_dev_test"
    dataset = load_dataset(DATASET_NAME)

    perplexity, per_item_ppl = eval_ppl(model, tokenizer, dataset["dev"])
    print(MODEL_NAME, ": Overall perplexity: ", perplexity)
    sentence_ppl_map = dict()
    for i, pipl in enumerate(per_item_ppl):
        sentence_ppl_map[dataset["dev"][i]["text"]] = pipl

    with open(f"{MODEL_NAME.replace('/', '-')}.ppl.out", "w") as f:
        json.dump(sentence_ppl_map, f)

main()
