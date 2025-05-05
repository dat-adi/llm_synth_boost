import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def calculate_perplexity(model, tokenizer, dataset):
    total_loss = 0.0
    total_tokens = 0

    print("Starting perplexity calculation...", flush=True)

    for idx, example in enumerate(dataset):
        text = example["text"]

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(model.device)

        # Calculate the model output
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss = outputs.loss
        num_tokens = input_ids.numel()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        # Print progress every 10 examples
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dataset)} examples...", flush=True)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def main():
    print("Starting the script...", flush=True)
                                                                                                                                                              
    model_name = sys.argv[1]                                                                                                                                  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                     
                                                                                                                                                              
    print(f"Loading tokenizer for model: {model_name}...", flush=True)                                                                                        
    tokenizer = AutoTokenizer.from_pretrained(model_name)                                                                                                     
    print("Tokenizer loaded successfully.", flush=True)                                                                                                       
                                                                                                                                                              
    print(f"Loading model: {model_name}...", flush=True)                                                                                                      
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")                                                    
    print("Model loaded successfully.", flush=True)                                                                                                           
                                                                                                                                                              
    model.to(device)                                                                                                                                          
    model.eval()                                                                                                                                              
                                                                                                                                                              
    print("Loading dataset...", flush=True)                                                                                                                   
    dataset = load_dataset("dogtooth/default_project_dev_test", split="dev_test")                                                                             
    print("Dataset loaded successfully. Starting evaluation...", flush=True)                                                                                  
                                                                                                                                                              
    # Calculate perplexity                                                                                                                                    
    perplexity = calculate_perplexity(model, tokenizer, dataset)                                                                                              
                                                                                                                                                              
    print(f"Perplexity: {perplexity}", flush=True)                                                                                                            
                                                                                                                                                              
if __name__ == "__main__":                                                                                                                                    
    main()                                                                                                                                                    
                                                                                                                                                              
                                                                                                                                            66,0-1        Bot 
