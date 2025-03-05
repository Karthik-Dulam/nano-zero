# %% [markdown]
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# %% [markdown]
# ### News

# %% [markdown]
# **Read our [blog post](https://unsloth.ai/blog/r1-reasoning) for guidance on how to train reasoning models.**
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# %% [markdown]
# ### Installation

# %%
 # %%capture
# !pip install unsloth vllm
# !pip install triton==3.1.0
# !pip install -U pynvml

# %% [markdown]
# ### Unsloth

# %% [markdown]
# Use `PatchFastRL` before all functions to patch GRPO and other RL algorithms!

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

# %% [markdown]
# Load up `Llama 3.1 8B Instruct`, and set parameters

# %%
from unsloth import is_bfloat16_supported
import torch
max_seq_length = 3*1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
    # device_map="cuda:1"
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# %% [markdown]
# ### Data Prep
# <a name="Data"></a>
# 
# We directly leverage [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data prep and all reward functions. You are free to create your own!

# %%
import re
from datasets import load_dataset, Dataset
import json
from datetime import datetime
from pathlib import Path

# Global dictionary to store rewards - using simple indices
response_rewards = {}

def log_response(prompts, completions, reward_name, rewards, answer=None, log_dir="response_logs"):
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"responses_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    # First reward function - initialize entries
    if reward_name == "xml_count":
        for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
            response_rewards[i] = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "completion": completion[0]['content'],
                "rewards": {reward_name: float(reward)}
            }
    # Last reward function - write logs and clear
    elif reward_name == "correctness":
        for i, reward in enumerate(rewards):
            if i in response_rewards:
                response_rewards[i]["rewards"][reward_name] = float(reward)
                # Add the answer to the log entry before writing
                if answer is not None and i < len(answer):
                    response_rewards[i]["expected_answer"] = answer[i]
                with open(log_file, "a") as f:
                    f.write(json.dumps(response_rewards[i]) + "\n")
        # Clear the dictionary after writing
        response_rewards.clear()
    # Middle reward functions - just add to existing entries
    else:
        for i, reward in enumerate(rewards):
            if i in response_rewards:
                response_rewards[i]["rewards"][reward_name] = float(reward)

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format, the answer tags must contain only the answer and no other text.
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def add_newlines(puzzle):
    puzzle = puzzle.split(" ")
    for i in range(9, 0, -1):
        puzzle.insert(i*9, "\n")
    return " ".join(puzzle).replace("\n ", "\n")

def get_sudoku(split = "train") -> Dataset:
    data = load_dataset(
        "json", 
        data_files="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/karthik/nano-zero/rl_data/sudoku_dataset_difficulty_0.1.json"
    )[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': "Solve this sudoku:\n" + add_newlines(x['puzzle'])}
        ],
        'answer': add_newlines(x['solution'])
    }) # type: ignore
    return data # type: ignore


dataset = get_sudoku()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [20*sum(i == j for i,j in zip(r,a))/len(a) for r,a in zip(extracted_responses, answer)]
    log_response(prompts, completions, "correctness", rewards, answer=answer)
    
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return rewards

def int_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [10*sum(char[-1].isdigit() for char in r.split(" ") if char)/len(r) if len(r) > 0 else 0 for r in extracted_responses]
    log_response(prompts, completions, "int_reward", rewards)
    return rewards

# def strict_format_reward_func(completions, prompts=None, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     rewards = [10 if match else 0.0 for match in matches]
#     log_response(prompts, completions, "strict_format", rewards)
#     return rewards

# def soft_format_reward_func(completions, prompts=None, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     rewards = [10 if match else 0.0 for match in matches]
#     log_response(prompts, completions, "soft_format", rewards)
#     return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 12.5
    if text.count("\n</reasoning>\n") == 1:
        count += 12.5
    if text.count("\n<answer>\n") == 1:
        count += 12.5
        count -= len(text.split("\n</answer>\n")[-1])*0.01
    if text.count("\n</answer>") == 1:
        count += 12.5
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.01
    return count

def xmlcount_reward_func(completions, prompts=None, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    log_response(prompts, completions, "xml_count", rewards)
    return rewards

# %% [markdown]
# <a name="Train"></a>
# ### Train the model
# 
# Now set up GRPO Trainer and all configurations!

# %%
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    beta=2,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 12,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 12, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = max_seq_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "tensorboard", # Can use Weights & Biases
    logging_dir="logs/LLAMA3.1-8B-Instruct-GRPO",
    output_dir = "outputs/LLAMA3.1-8B-Instruct-GRPO",
    temperature=0.6,
    # resume_from_checkpoint=True,
)

# %% [markdown]
# And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!
# 
# You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!
# 
# | Step | Training Loss | reward    | reward_std | completion_length | kl       |
# |------|---------------|-----------|------------|-------------------|----------|
# | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
# | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
# | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
# 

# %%
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        # soft_format_reward_func,
        # strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# %% [markdown]
# <a name="Inference"></a>
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:

# %%
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.5,
    top_p = 0.8,
    max_tokens = max_seq_length,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

output

# %% [markdown]
# And now with the LoRA we just trained with GRPO - we first save the LoRA first!

# %%
model.save_lora("grpo_saved_lora")

# %% [markdown]
# Now we load the LoRA and test:

# %%
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = max_seq_length,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

output

# %% [markdown]
# Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

# %% [markdown]
# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# %%
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# %% [markdown]
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)

# %%
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )

# %% [markdown]
# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Llama 3.2 Conversational notebook. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
#


