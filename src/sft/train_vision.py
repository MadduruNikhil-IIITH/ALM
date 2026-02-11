"""
Phase 1: Vision Model SFT (Image → UPG JSON)
PERMANENT STABLE VERSION – batched=False + single-example preprocessing
No batching bugs, no grid_thw unpack error, no pixel_values length mismatch, no serialization crash
CUDA-ready with device_map="auto" + bf16
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from PIL import Image

# ────────────────────────────────────────────────
# CONFIG – minimal for first CUDA test run
# ────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"   # 2B = fast (~6–8 GB VRAM)
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

BATCH_SIZE = 1                              # must be 1 with batched=False
GRAD_ACCUM = 8                              # effective batch size = 8
EPOCHS = 1                                  # 1 epoch for quick test
LR = 5e-5                                   # safe to avoid NaN

PAIRS_FILE = Path("data/processed/ai2d_vision_sft_pairs.json")
OUTPUT_DIR = Path("checkpoints/vision_sft_test")
MODEL_SAVE_DIR = Path("models/vision_parser_sft_test")

# ────────────────────────────────────────────────
# Load pairs – keep upg_target as string
# ────────────────────────────────────────────────
print("Loading pairs...")
with open(PAIRS_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} pairs")

# Take minimum data for first CUDA test
MIN_DATA = 20                               # ← very small for first test (increase later)
pairs = pairs[:MIN_DATA]
print(f"Using {len(pairs)} examples for this test run")

# Simple Dataset – only strings & paths (no nested dicts)
dataset_dict = {
    "image_path": [p["image_path"] for p in pairs],
    "upg_target": [p["upg_target"] for p in pairs],  # already json string
    "diagram_id": [p["diagram_id"] for p in pairs]
}
dataset = Dataset.from_dict(dataset_dict)

# ────────────────────────────────────────────────
# Processor & Model – CUDA starts here
# ────────────────────────────────────────────────
print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",                  # ← automatic CUDA placement
    low_cpu_mem_usage=True
)

print("Model device:", next(model.parameters()).device)  # should be cuda:0

# LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ────────────────────────────────────────────────
# Preprocessing – single-example only (batched=False)
# ────────────────────────────────────────────────
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    tgt_str = example["upg_target"]

    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Parse this physics diagram into the Unified Physical Graph JSON format."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": tgt_str}]
        }
    ]

    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=2048
    )

    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100

    return inputs

# Apply preprocessing – batched=False is the permanent fix
print("Preprocessing dataset (single-example mode)...")
train_dataset = dataset.map(preprocess, batched=False, remove_columns=dataset.column_names)

# ────────────────────────────────────────────────
# Training Arguments – CUDA accelerated
# ────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=20,
    bf16=True,
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    report_to="tensorboard",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    lr_scheduler_type="cosine"
)

# ────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

# Start training
print("Starting SFT training on CUDA...")
trainer.train()

# Save
trainer.save_model(str(MODEL_SAVE_DIR))
processor.save_pretrained(str(MODEL_SAVE_DIR))

print(f"Training finished.")
print(f"Model saved to: {MODEL_SAVE_DIR}")
print(f"Checkpoints: {OUTPUT_DIR}")
print("View loss: tensorboard --logdir checkpoints/vision_sft_test")