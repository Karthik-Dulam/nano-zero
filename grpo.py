# -*- coding: utf-8 -*-
"""
Qwen GRPO Training for Sudoku

This script performs GRPO training on a Sudoku dataset.
It uses the Qwen2.5-Math-1.5B-Instruct model and trains on sudoku puzzles, expecting the dataset
in a JSON file (sudoku_sft_data.json) with each sample containing 'instruction',
'input', and 'output' fields. The output should include a chain-of-thought enclosed
in <thonk>...</thonk> and the final answer in <ans>...</ans>.
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import os

# Define the system prompt for Sudoku tasks
SYSTEM_PROMPT = """Respond in the following format:
<thonk>
...your reasoning...
</thonk>
<ans>
...final answer...
</ans>
"""


def extract_ans(text: str) -> str:
    """Extract the answer text from the model output using <ans> tags."""
    try:
        answer = text.split("<ans>")[-1].split("</ans>")[0]
    except IndexError:
        return ""
    return answer.strip()


def get_sudoku_dataset():
    """
    Load and preprocess the Sudoku dataset from a JSON file.
    The JSON file should contain a list of dictionaries with keys:
    'instruction', 'input', and 'output'.
    Returns a Dataset with 'prompt' and 'answer' fields.
    """
    dataset = load_dataset("json", data_files="sudoku_sft_data.json")["train"]
    
    def preprocess(sample):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{sample['instruction']}\nPuzzle: {sample['input']}"}
            ],
            "answer": extract_ans(sample["output"])
        }
    
    return dataset.map(preprocess)


# Reward function that checks for correctness by comparing the extracted answer with the ground truth.
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    # Debug output for one example
    print("==== Example Debug Info ====")
    print("Prompt:", prompts[0][-1]['content'])
    print("Ground Truth Answer:", answer[0])
    print("Model Response:", responses[0])
    extracted_responses = [extract_ans(r) for r in responses]
    print("Extracted Answer:", extracted_responses[0])
    return [2.0 if ext == gt else 0.0 for ext, gt in zip(extracted_responses, answer)]


# Reward function that checks if the output is in the proper format (contains both <thonk> and <ans> tags).
def format_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    def is_proper_format(text: str) -> bool:
        return "<thonk>" in text and "</thonk>" in text and "<ans>" in text and "</ans>" in text
    return [0.5 if is_proper_format(r) else 0.0 for r in responses]


# Define GRPO training arguments
training_args = GRPOConfig(
    output_dir="outputs/Qwen2.5-0.5B-GRPO-Sudoku",
    run_name="Qwen2.5-0.5B-GRPO-Sudoku",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=1.0,
    log_on_each_node=False,
    report_to="none",  # Disabling Wandb logging
    gradient_checkpointing=True,
)

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_checkpoint = "sft_output/epoch=0-step=11250.ckpt"
# checkpoint = torch.load(model_checkpoint)
# model_state_dict = {}
# for key in checkpoint["state_dict"].keys():
#     if "model" in key:
#         model_state_dict[key.replace("model.", "", 1)] = checkpoint["state_dict"][key]

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Important for autoregressive models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize the GRPO trainer with our Sudoku dataset and reward functions
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward_func, correctness_reward_func],
    args=training_args,
    train_dataset=get_sudoku_dataset(),
)


if __name__ == "__main__":
    print("Starting GRPO training on Sudoku dataset...")
    trainer.train()
    print("GRPO training completed!")
