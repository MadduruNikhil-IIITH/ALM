"""
Phase 1: Vision Model SFT (Image → UPG JSON)
STABLE VERSION for RTX 4060 8GB + wandb logging with curves
- Frequent logging for visible loss curves
- Eval temporarily disabled to avoid token mismatch
- Processor truncation forced off
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from PIL import Image
import wandb

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 1
LR = 5e-5

PAIRS_FILE = Path("data/processed/ai2d_vision_sft_pairs.json")
OUTPUT_DIR = Path("checkpoints/vision_sft_test")
MODEL_SAVE_DIR = Path("models/vision_parser_sft_test")

# ────────────────────────────────────────────────
# Load data (small for testing)
# ────────────────────────────────────────────────
print("Loading pairs...")
with open(PAIRS_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

MIN_DATA = 100                               # Increase later
pairs = pairs[:MIN_DATA]
print(f"Using {len(pairs)} examples")

wandb.init(
    project="ALM",          # Your project name
    name=f"vision_test_sft_ai2d_{MIN_DATA}ex",        # Run name
    config={                            # Optional: log hyperparameters
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "batch_size": BATCH_SIZE * GRAD_ACCUM,
        "epochs": EPOCHS,
        "dataset_size": len(pairs),
        "min_data": MIN_DATA,
    },
    # Optional: tags for filtering
    tags=["phase1", "vision-parser", "ai2d", "qwen2-vl-2b"]
)


def format_example(example):
    image = Image.open(example["image_path"]).convert("RGB")
    tgt_str = example["upg_target"]
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Parse this physics diagram into the Unified Physical Graph JSON format."}]},
        {"role": "assistant", "content": [{"type": "text", "text": tgt_str}]}
    ]
    return {"images": [image], "messages": messages}

dataset = Dataset.from_dict({
    "image_path": [p["image_path"] for p in pairs],
    "upg_target": [p["upg_target"] for p in pairs],
    "diagram_id": [p["diagram_id"] for p in pairs]
})
train_dataset = dataset.map(format_example, batched=False, remove_columns=dataset.column_names)
print(f"Train examples: {len(train_dataset)}")  # No eval for now

# ────────────────────────────────────────────────
# Processor & Model
# ────────────────────────────────────────────────
print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Critical fix for mismatch: Force-disable truncation
processor.tokenizer.truncation = False
processor.tokenizer.padding = False

# Cap resolution for 8GB VRAM
processor.image_processor.size = {"shortest_edge": 224, "longest_edge": 896}

# Long context support
processor.tokenizer.model_max_length = 65536  # Higher to give headroom

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Model device:", next(model.parameters()).device)

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
# SFTConfig
# ────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=20,
    bf16=True,
    logging_steps=1,                        # Frequent for curves
    save_steps=5,
    save_total_limit=2,
    report_to="wandb",
    dataloader_num_workers=0,
    lr_scheduler_type="cosine",

    # VRAM savings
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Eval disabled temporarily to avoid mismatch
    # eval_strategy="steps",
    # eval_steps=1,
    # do_eval=True,
)

# ────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,  # ← Commented out
    processing_class=processor,
)

print("Starting SFT training on RTX 4060 8GB with wandb logging...")
trainer.train()

# Save
trainer.save_model(str(MODEL_SAVE_DIR))
processor.save_pretrained(str(MODEL_SAVE_DIR))

print(f"Training finished.")
print(f"Model saved to: {MODEL_SAVE_DIR}")