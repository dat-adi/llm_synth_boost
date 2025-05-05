import torch
import math

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, Trainer,
    TrainingArguments, TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

def set_up_model_for_qlora(model):
    """Set up the model for QLoRA"""

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # important for QLoRA
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model

def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    inputs["labels"] = inputs["input_ids"].copy()   # <- THIS LINE is critical
    return inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert to torch tensors
    logits = torch.from_numpy(logits).float().to(device)  # (batch_size, seq_len, vocab_size)
    labels = torch.from_numpy(labels).to(device)          # (batch_size, seq_len)

    # Shift so that tokens <n> predict token <n+1>
    shift_logits = logits[..., :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()      # (batch_size, seq_len-1)

    # Flatten for loss computation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (total_tokens, vocab_size)
    shift_labels = shift_labels.view(-1)                         # (total_tokens)

    # Compute CrossEntropyLoss (ignoring pad tokens)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits, shift_labels)

    # Number of non-pad tokens
    non_pad_tokens = (shift_labels != tokenizer.pad_token_id).sum()

    avg_loss = (loss / non_pad_tokens).item()
    perplexity = math.exp(avg_loss)

    return {
        'eval_loss': avg_loss,
        'perplexity': perplexity,
    }

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Step {state.global_step} (Eval): {metrics}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step} (Train): {logs}")

def evaluate_dataset(tokenizer, model, tokenized_eval_dataset):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False   # causal LM = next-token prediction
    )

    eval_loader = DataLoader(
        tokenized_eval_dataset, batch_size=1,
        shuffle=False, collate_fn=collator
    )

    model.eval()
    total_loss = total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)            # labels already included
            loss = outputs.loss                 # mean NLL per token
            token_count = batch["attention_mask"].sum().item()
            total_loss  += loss.item() * token_count
            total_tokens += token_count

    avg_loss  = total_loss / total_tokens        # nats per token
    perplexity = math.exp(avg_loss)
    print(f"avg_loss={avg_loss:.4f}  perplexity={perplexity:.2f}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model to fine-tune
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)
model = set_up_model_for_qlora(model)

# Load datasets
train_dataset = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=42).select(range(5000))
eval_dataset = load_dataset("dogtooth/default_project_dev_test", split="dev", keep_in_memory=False)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    # fp16_full_eval=True,
    # prediction_loss_only=True, # MESSED UP EPOCH WISE PPL CALC
    logging_steps=50,
    save_steps=30,
    eval_strategy="steps",    # (spelling)
    eval_steps=30,
    eval_accumulation_steps=4,     # optional: streams loss to CPU to save VRAM
    save_total_limit=1,
    report_to="none",
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
evaluate_dataset(tokenizer, model, tokenized_eval_dataset)

test_dataset = load_dataset("dogtooth/default_project_dev_test", split="dev_test", keep_in_memory=False)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
evaluate_dataset(tokenizer, model, tokenized_test_dataset)
