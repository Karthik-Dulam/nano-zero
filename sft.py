import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import os

# Hyperparameters
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_PATH = "sudoku_sft_data.json"
OUTPUT_DIR = "sft_output"
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 3

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token



# Load SFT dataset
def load_sft_data(path):
    try:
        data = load_dataset("json", data_files=path)
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return None


sft_data = load_sft_data(DATASET_PATH)

if sft_data is None:
    exit()


# Preprocess data
def preprocess_function(examples):
    inputs = examples["instruction"] + examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=2**13, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, return_tensors="pt", padding="max_length", max_length=2**13, truncation=True)
    model_inputs["labels"] = labels["input_ids"].masked_fill(
        labels["input_ids"] == tokenizer.pad_token_id, -100
    )
    return model_inputs


tokenized_datasets = sft_data.map(preprocess_function)

print(tokenized_datasets["train"][0])

# Split the dataset into training and validation sets
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,  # Save only the last 2 checkpoints
    push_to_hub=False,  # Set to True if you want to push to the Hugging Face Hub
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    eval_steps=10,  # Evaluate every 10 steps
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Training
trainer.train()

# Saving the fine-tuned model
trainer.save_model()
