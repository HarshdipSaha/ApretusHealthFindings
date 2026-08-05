import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Setup & Reproducibility
seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_id = "swiss-ai/Apertus-8B-Instruct-2509"
dataset_id = "lavita/ChatDoctor-HealthCareMagic-100k"
output_dir = "./apertus-medical-lora"

# 2. Dataset Preprocessing
print("Loading dataset...")
dataset = load_dataset(dataset_id, split="train")

def format_instruction(sample):
    return {"text": f"Patient: {sample['input']}\nDoctor: {sample['output']}"}

print("Preprocessing dataset...")
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.01)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 3. Model Loading (QLoRA)
print("Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 4. LoRA Configuration
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Training Arguments (using SFTConfig for TRL 1.5.0)
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=500, 
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    optim="paged_adamw_32bit",
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    max_length=512,
    dataset_text_field="text",
)

# 6. Execution
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=sft_config,
)

print("Starting training...")
trainer.train()

print(f"Saving fine-tuned adapters to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Training complete!")
