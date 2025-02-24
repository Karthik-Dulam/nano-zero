import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

# Hyperparameters

def get_args():
    parser = argparse.ArgumentParser(description='Train a model using SFT')
    
    # Model and data parameters
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                      help='Name or path of the base model')
    parser.add_argument('--dataset_path', type=str, default="sudoku_sft_data.json",
                      help='Path to the training dataset')
    parser.add_argument('--output_dir', type=str, default="outputs/Qwen2.5-0.5B-SFT",
                      help='Directory to save the model outputs')
    
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Number of steps for gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=1e-7,
                      help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    parser.add_argument('--log_every_n_steps', type=int, default=30000,
                      help='Log every N steps')
    parser.add_argument('--val_check_interval', type=int, default=30000,
                      help='Run validation every N steps')
    parser.add_argument('--save_every_n_steps', type=int, default=30000,
                      help='Save checkpoint every N steps')
    
    parser.add_argument('--devices', nargs='+', type=int, default=[0],
                      help='List of GPU devices to use')
    parser.add_argument('--no_checkpointing', action='store_false', dest='checkpointing',
                      help='Disable gradient checkpointing')
    parser.set_defaults(checkpointing=True)
    
    args = parser.parse_args()
    return args

def get_model_and_tokenizer(model_name, checkpointing):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_sft_data(path):
    try:
        data = load_dataset("json", data_files=path)
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return None

def preprocess_function(examples, tokenizer, output=True):
    out = f"\n{examples['output']}\n</output>{tokenizer.eos_token}" if output else ""
    prompt = (
        f"<instruction>\n{examples['instruction']}\n</instruction>\n"
        f"<input>\n{examples['input']}\n</input>\n"
        f"<output>{out}"
    )
    
    return tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        return_attention_mask=False,
    )

def prepare_dataloaders(tokenized_datasets, batch_size):
    train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader

class SFTModel(pl.LightningModule):
    def __init__(self, model, learning_rate, batch_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
                 batch_size=self.batch_size,
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
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size,
                 sync_dist=True)
        return loss

def get_trainer(args):
    logger = TensorBoardLogger("tb_logs", name="Qwen2.5-0.5B-SFT")
    
    return pl.Trainer(
        max_epochs=args.num_epochs,
        devices=args.devices,
        strategy="ddp",
        precision="bf16-mixed",
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.output_dir,
                filename="sft-v{version}-{epoch}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
                save_last=True,
                every_n_train_steps=args.save_every_n_steps,
                auto_insert_metric_name=False
            )
        ]
    )

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    
    # Update global constants with parsed arguments
    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    GRADIENT_CLIP = args.gradient_clip
    LOG_EVERY_N_STEPS = args.log_every_n_steps
    VAL_CHECK_INTERVAL = args.val_check_interval
    SAVE_EVERY_N_STEPS = args.save_every_n_steps
    DEVICES = args.devices
    CHECKPOINTING = args.checkpointing
    
    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME, CHECKPOINTING)
    
    # Load and preprocess data
    sft_data = load_sft_data(DATASET_PATH)
    if sft_data is None:
        exit()
    
    # Preprocess the dataset
    tokenized_datasets = sft_data.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=["instruction", "input", "output"]
    ).with_format("torch")
    
    # Prepare dataloaders
    train_loader, eval_loader = prepare_dataloaders(tokenized_datasets, BATCH_SIZE)
    
    # Initialize the model and trainer
    sft_model = SFTModel(model, LEARNING_RATE, BATCH_SIZE)
    trainer = get_trainer(args)
    
    # Train the model
    trainer.fit(sft_model, train_loader, eval_loader)
    
    # Save the final model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
