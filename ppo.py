import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import json
import os

# Hyperparameters
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_PATH = "path/to/sudoku_dataset.json"
NUM_EPOCHS = 100

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Reward function
def calculate_reward(response: str, correct_answer: str) -> float:
    """Calculate reward based on correct answer and formatting."""
    # Extract answer between tags
    start_tag = "<ans>"
    end_tag = "</ans>"
    
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        return 0.0
    
    extracted = response[start_idx+len(start_tag):end_idx].strip()
    
    # Normalize extracted answer
    digits = "".join(filter(str.isdigit, extracted))
    if len(digits) != 81:
        return 0.0
    
    # Compare with correct answer
    return 1.0 if digits == correct_answer else 0.0

# Load dataset
def format_prompt(puzzle: str) -> str:
    return f"<instruction>Solve this Sudoku puzzle:\n{puzzle}</instruction>"

dataset = Dataset.from_json(DATASET_PATH).map(lambda x: {
    "prompt": format_prompt(x["puzzle"]),
    "correct_answer": x["solution"]
})

# PPO Configuration
ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1e-5,
    remove_unused_columns=False,
    log_with="wandb",  # Enable logging with Weights & Biases
)

# Initialize trainer
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=dataset,
)

# Generation settings
gen_kwargs = {
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch in ppo_trainer.dataloader:
        # Get batch data
        prompts = batch["prompt"]
        correct_answers = batch["correct_answer"]
        
        # Tokenize prompts
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            inputs.input_ids.to(ppo_trainer.accelerator.device), 
            **gen_kwargs
        )
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Calculate rewards
        rewards = [
            torch.tensor(calculate_reward(resp, ans)).to(ppo_trainer.accelerator.device) 
            for resp, ans in zip(responses, correct_answers)
        ]
        
        # Run PPO step
        stats = ppo_trainer.step(
            [inputs.input_ids], 
            [response_tensors], 
            rewards
        )
        
        # Log progress
        ppo_trainer.log_stats(stats, batch, rewards)
