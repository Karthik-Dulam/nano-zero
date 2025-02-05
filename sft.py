import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Hyperparameters
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "sudoku_sft_data.json"
OUTPUT_DIR = "outputs/Qwen2.5-0.5B-SFT"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
GRADIENT_CLIP = 1.0
LOG_EVERY_N_STEPS = 30000  # Log every 100 steps
VAL_CHECK_INTERVAL = 30000  # Validate every 200 steps
SAVE_EVERY_N_STEPS = 30000

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

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

def preprocess_function(examples):
    prompt = (
        f"<instruction>\n{examples['instruction']}\n</instruction>\n"
        f"<input>\n{examples['input']}\n</input>\n"
        f"<output>\n{examples['output']}\n</output>{tokenizer.eos_token}"
    )
    
    return tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        return_attention_mask=False,
    )

tokenized_datasets = sft_data.map(
    preprocess_function, remove_columns=["instruction", "input", "output"]
).with_format("torch")

train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)


class SFTModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.automatic_optimization = True

    def training_step(self, batch, batch_idx):
        self.model.train()
        input_ids = batch["input_ids"].squeeze(1)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        self.log("train_loss", loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True,
                 batch_size=BATCH_SIZE,
                 sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids = batch["input_ids"].squeeze(1)
        loss = self.model(input_ids, labels=input_ids).loss
        self.log("val_loss", loss, 
                 on_step=False,  # Only log at end of validation
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=BATCH_SIZE,
                 sync_dist=True)
        return loss


sft_model = SFTModel(model, LEARNING_RATE)

logger = TensorBoardLogger("tb_logs", name="Qwen2.5-0.5B-SFT")

trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    devices=[0],
    strategy="ddp",
    precision="bf16-mixed",
    gradient_clip_val=GRADIENT_CLIP,
    accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    val_check_interval=VAL_CHECK_INTERVAL,
    check_val_every_n_epoch=None,  # Disable epoch-based validation
    logger=logger,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            filename="sft-v{version}-{epoch}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_train_steps=SAVE_EVERY_N_STEPS,
            auto_insert_metric_name=False
        )
    ]
)
trainer.fit(sft_model, train_loader, eval_loader)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
