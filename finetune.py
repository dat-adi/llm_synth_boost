import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and tokenizer setup
model_id = "meta-llama/Llama-3.1-8B"

# Quantization config - using 8-bit to fit in 16GB RAM
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for training with LoRA
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration optimized for 16GB RAM
lora_config = LoraConfig(
    r=8,  # Lower rank for memory efficiency
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Load datasets
train_dataset = load_dataset("JeanKaddour/minipile", split="train")
# train_dataset = load_dataset("csv", data_files="./data/synthetic_all.csv", split="train")
eval_dataset = load_dataset("dogtooth/default_project_dev_test", split="dev")

# Sample 5000 examples from training dataset
train_dataset = train_dataset.shuffle(seed=42).select(range(1000)) # pre-shuffled
# train_dataset = train_dataset.select(range(250))

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,  # Reduced context length for memory efficiency
        padding="max_length",
    )

# Tokenize datasets
tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=train_dataset.column_names
)
tokenized_eval = eval_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=eval_dataset.column_names
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Small batch size for memory efficiency
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
    logging_steps=50,  # Log every 50 steps
    logging_dir="./logs",
    report_to="tensorboard",
    save_total_limit=2,  # Save only the last 2 checkpoints
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Training with evaluation at each epoch
print("Starting training...")
train_result = trainer.train()

# Get perplexity scores
epochs = list(range(1, 4))  # 3 epochs
perplexities = []

# Calculate perplexity for each epoch
for epoch in epochs:
    checkpoint_dir = f"./results/checkpoint-{epoch * len(tokenized_train) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps}"
    if os.path.exists(checkpoint_dir):
        # Load model from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
        # Evaluate
        eval_results = trainer.evaluate()
        perplexity = np.exp(eval_results["eval_loss"])
        perplexities.append(perplexity)
        print(f"Epoch {epoch}: Perplexity = {perplexity:.4f}")
    else:
        print(f"Checkpoint for epoch {epoch} not found.")

try:
    # Plot perplexity vs epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexities, marker='o')
    plt.title('Validation Perplexity vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig('perplexity_plot.png')
    plt.show()
except Exception as err:
    print(err)

# Print training metrics
print("Training completed!")
print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Training throughput: {train_result.metrics['train_samples_per_second']:.2f} samples/s")

# Save the final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("Model saved to ./final_model")

# Plot training loss from logs
from tensorboard.backend.event_processing import event_accumulator
import glob

# Find the latest log file
log_files = glob.glob("./logs/events.out.tfevents.*")
if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    
    # Load the events
    ea = event_accumulator.EventAccumulator(latest_log)
    ea.Reload()
    
    # Extract the training loss
    if 'train/loss' in ea.scalars.Keys():
        try:
            loss_events = ea.scalars.Items('train/loss')
            steps = [event.step for event in loss_events]
            losses = [event.value for event in loss_events]
            
            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.title('Training Loss vs. Steps')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('training_loss_plot.png')
            plt.show()
        except Exception as err:
            print("Could not retrieve training loss: ", err)

    # Extract the evaluation loss
    if 'eval/loss' in ea.scalars.Keys():
        try:
            loss_events = ea.scalars.Items('eval/loss')
            steps = [event.step for event in loss_events]
            losses = [event.value for event in loss_events]
            
            # Plot eval loss
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.title('Evaluation Loss vs. Steps')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('eval_loss_plot.png')
            plt.show()
        except Exception as err:
            print("Could not retrieve training loss: ", err)
