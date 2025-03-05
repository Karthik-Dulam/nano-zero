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
from peft import LoraConfig, get_peft_model

# Define the system prompt for Sudoku tasks


def extract_ans(text: str) -> str:
    """Extract the answer text from the model output using <ans> tags."""
    try:
        answer = text.split("<ans>")[1].split("</ans>")[0]
    except IndexError:
        return ""
    return answer.strip()


def get_sudoku_dataset():
    """
    Load and preprocess the Sudoku dataset from the rl_data folder.
    Each sample contains 'puzzle' and 'solution' fields.
    Returns a Dataset with 'prompt' and 'answer' fields.
    """
    # Load all JSON files in the rl_data directory
    dataset = load_dataset("json", data_files="rl_data/*0.1.json")["train"]

    def preprocess(sample):
        instruction = (
            "Solve this Sudoku puzzle, do your thinking in <thinking> </thinking> tags and write your answer in <ans> </ans> tags in the same format as the input:"
            "\nFor example: <thinking> Notice that the only number missing in row 8... so the answer must be 9 3 4 7 1 6 2 8 5... </thinking>\n<ans> 9 3 4 7 1 6 2 8 5... </ans><|endoftext|>"
        )

        # Format the prompt exactly as in sft.py
        prompt_text = (
            f"<instruction>\n{instruction}\n</instruction>\n"
            f"<input>\n{sample['puzzle']}\n</input>\n"
            f"<output>\n<thinking>"
        )

        return {"prompt": prompt_text, "answer": sample["solution"]}

    return dataset.map(preprocess)


# Simplified correctness reward that just checks exact match
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Debug output for one example
    print("==== Example Debug Info ====")
    print("Prompt:", prompts[0])
    # print("Model Response:", completions[0])
    extracted_responses = [extract_ans(r) for r in completions]
    print(f"Completions: {completions}")
    print("Extracted Answers:", extracted_responses)
    print("Ground Truth Answer:", answer[0])

    rewards = []
    for ext, gt in zip(extracted_responses, answer):
        # Remove whitespace and compare directly
        ext = ''.join(ext.split())
        gt = ''.join(gt.split())
        # Reward based on number of matching characters
        rewards.append(sum([3.0 if a == b else 0.0 for a, b in zip(ext, gt)]) / len(gt))
    print ("Correctness Rewards:", rewards)
    print("=============================")

    
    return rewards


# Simplified format reward that just checks for required tags
def format_reward_func(completions, **kwargs) -> list[float]:
    def has_required_tags(text: str) -> bool:
        required_tags = ['<thinking>', '</thinking>', '<ans>', '</ans><|endoftext|>']
        return all(tag in text for tag in required_tags)

    return [0.5 if has_required_tags(r) else 0.0 for r in completions]


# Simplified XML count reward
def count_xml(text, **kwargs) -> float:
    score = 0.0
    # Award points for correct tag placement
    if '<thinking>' in text and '</thinking>' in text:
        score += 0.25
    if '<ans>' in text and '</ans><|endoftext|>' in text:
        score += 0.25
    # Penalize multiple tag pairs
    score -= 0.1 * (text.count('<thinking>') - 1)
    score -= 0.1 * (text.count('<ans>') - 1)
    return max(-1.0, score) 

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]


# Modify training arguments to remove FSDP
training_args = GRPOConfig(
    # output_dir="outputs/SmolLM-360M-Instruct-Sudoku",
    # run_name="SmolLM-360M-Instruct-Sudoku",
    output_dir="outputs/Qwen2.5-0.5B-GRPO-Sudoku",
    run_name="Qwen2.5-0.5B-GRPO-Sudoku",
    learning_rate=3e-4,
    adam_beta1=0.9,
    adam_beta2=0.99,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=False,
    fp16=False,
    per_device_train_batch_size=4,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=5000,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.5,
    log_on_each_node=False,
    temperature=0.5,
    report_to="none",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch",
    # optim="galore_adamw",
    # optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    # optim_args="rank=64, update_proj_gap=100, scale=0.10",
    torch_empty_cache_steps=1,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.2,
    vllm_device="cuda:0",
    vllm_dtype=torch.float32,
)

# Define LoRA Config for PEFT
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
)

# Load the model and tokenizer
model_name = "outputs/Qwen2.5-0.5B-SFT"
# model_name = "HuggingFaceTB/SmolLM-360M-Instruct"

# Modify model loading for model parallelism
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # Changed for automatic model parallelism
    torch_dtype=torch.float32,
).to("cuda:1")

# Prepare model for PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


model.config.use_cache = False  # Disable cache to prevent gradient issues

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Crucial for autoregressive models
tokenizer.truncation_side = "left"


# Initialize the GRPO trainer with our Sudoku dataset and reward functions
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        correctness_reward_func,
        xmlcount_reward_func,
    ],
    args=training_args,
    train_dataset=get_sudoku_dataset(),
)


if __name__ == "__main__":
    # Remove manual distributed initialization
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"Starting GRPO training on Sudoku dataset (rank {local_rank})...")
    trainer.train()

    if local_rank == 0:
        print("GRPO training completed!")
        trainer.save_model("final_model")
