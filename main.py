from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from datasets import load_dataset


def calculate_perplexity(model, tokenizer, dataset):
    total_loss = 0
    total_tokens = 0
    
    for item in dataset:
        text = item["text"]  # Assuming the dataset has a "text" field
        
        # Tokenize and prepare inputs
        encodings = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = encodings.input_ids
        
        # Setup targets (shifted input_ids)
        with torch.no_grad():
            outputs = model(input_ids[:, :-1], labels=input_ids[:, 1:])
            
        # Get the loss
        neg_log_likelihood = outputs.loss
        
        # Accumulate loss and token count
        total_loss += neg_log_likelihood.item() * (input_ids.size(1) - 1)
        total_tokens += input_ids.size(1) - 1
    
    # Calculate perplexity across the entire dataset
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    
    return ppl.item()

def perplex_eval_sample_texts():
    # Sample texts for perplexity evaluation

    # 1. Simple narrative text
    simple_narrative = """
    The sun rose over the mountains, casting long shadows across the valley. Birds began to sing their morning songs as the first rays of light touched the dew-covered grass. A small fox emerged from its den, sniffing the cool morning air before setting off to hunt for breakfast.
    """

    # 2. Technical explanation
    technical_explanation = """
    Transformer models utilize self-attention mechanisms to process sequential data. Unlike recurrent neural networks, transformers process the entire sequence simultaneously, which enables more efficient parallel computation. The architecture consists of encoder and decoder blocks, each containing multi-head attention layers and feed-forward neural networks. This design has revolutionized natural language processing by capturing long-range dependencies more effectively than previous approaches.
    """

    # 3. Abstract reasoning
    abstract_reasoning = """
    The relationship between consciousness and physical reality remains one of philosophy's most enduring questions. While materialists argue that consciousness emerges from physical processes in the brain, dualists maintain that mental phenomena cannot be reduced to physical explanations. Recent developments in quantum physics have introduced additional complexity to this debate, suggesting potential connections between observation, measurement, and the nature of reality itself.
    """

    # 4. Code-like content
    code_content = """
    def implement_attention(query, key, value):
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        # Scale scores
        scores = scores / math.sqrt(key.size(-1))
        # Apply softmax to normalize
        attention_weights = F.softmax(scores, dim=-1)
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    """

    # 5. Creative/poetic content
    creative_content = """
    Midnight whispers secrets to the ancient trees. Stars, like scattered diamonds, punctuate the darkness overhead. Time feels suspended in this moment, neither rushing forward nor dwelling in the past. The world breathes slowly, deliberately, as if gathering strength for tomorrow's dawn.
    """

    # Combine samples or use individually
    evaluation_texts = [simple_narrative, technical_explanation, abstract_reasoning, code_content, creative_content]

    # Example usage
    for idx, text in enumerate(evaluation_texts):
        print(f"Sample {idx+1} Perplexity:", calculate_perplexity(model, tokenizer, text))

def main():
    parser = argparse.ArgumentParser(description="Script that takes a model name")
    parser.add_argument("MODEL_NAME", type=str, help="Name of the model to use")

    args = parser.parse_args()
    MODEL_NAME = args.MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    DATASET_NAME = "dogtooth/default_project_dev_test"
    dataset = load_dataset(DATASET_NAME)

    ppl = calculate_perplexity(model, tokenizer, dataset["dev"])
    print(MODEL_NAME, " perplexity: ", ppl)

main()
