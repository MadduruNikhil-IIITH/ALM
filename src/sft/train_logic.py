"""
Phase 2: Logic Grounding SFT Training with Weights & Biases (W&B) integration
Trains a model to generate Chain-of-Physical-Thought (CoPT) from ScienceQA physics examples.

Run:
    python src/sft/train_logic.py
"""
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import wandb


# ────────────────────────────────────────────────────────────────
# Placeholder Prompt (v1)
# ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise middle-school / high-school physics reasoning assistant.
Your task is to answer the question using ONLY the information provided in the lecture.
Follow this exact structure in your response:

1. Core Concept: Restate the most important physics idea from the lecture in 1-2 sentences.
2. Key Observations: List the important facts or conditions given in the question.
3. Step-by-step Reasoning: Explain your thinking clearly, step by step.
4. Answer: Write only one line:
   Final Answer: [the exact text of the correct choice]

Do NOT add extra explanations after the Final Answer line.
Do NOT use information you know from outside the lecture.

Question: {question}

Lecture: {lecture}

Choices:
{choices}

Think step by step inside steps 1–3."""


def format_example(example):
    """Format one example into prompt + completion for SFT."""
    choices_str = "\n".join(
        f"{i}. {choice.strip()}" for i, choice in enumerate(example["choices"])
    )

    prompt = SYSTEM_PROMPT.format(
        question=example["question"].strip(),
        lecture=example["lecture"].strip(),
        choices=choices_str
    )

    solution = example["solution"].strip()
    if not solution.endswith("."):
        solution += "."

    correct_choice_text = example["choices"][example["answer"]].strip()
    target = f"{solution}\n\nFinal Answer: {correct_choice_text}"

    full_text = prompt + "\n\n" + target

    return {"text": full_text}


def main():
    # ────────────────────────────────────────────────────────────────
    # Configuration
    # ────────────────────────────────────────────────────────────────
    MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
    OUTPUT_DIR = "output/logic_sft_qwen2-1.5b"
    MAX_SEQ_LENGTH = 1024
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.03
    LOGGING_STEPS = 25
    EVAL_STEPS = 200
    SAVE_STEPS = 400
    FP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    DATA_DIR = Path("data/processed")
    train_path = DATA_DIR / "scienceqa_physics_train.json"
    val_path = DATA_DIR / "scienceqa_physics_val.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing split files:\n{train_path}\n{val_path}")

    # ────────────────────────────────────────────────────────────────
    # Weights & Biases Setup (NEW)
    # ────────────────────────────────────────────────────────────────
    wandb_project = "alm-logic-sft"              # Project name in W&B dashboard
    wandb_run_name = f"qwen2-1.5b-sft-ep{NUM_EPOCHS}-lr{LEARNING_RATE}"  # Unique run name

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model": MODEL_NAME,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "lora_rank": 64,
            "dataset": "scienceqa_physics_filtered",
            "max_seq_length": MAX_SEQ_LENGTH,
        },
        tags=["phase2", "physics", "sft", "qwen2"],
        notes="Phase 2 logic grounding training on filtered ScienceQA physics subset"
    )

    # ────────────────────────────────────────────────────────────────
    # Quantization & Model Loading
    # ────────────────────────────────────────────────────────────────
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=FP_DTYPE,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=FP_DTYPE,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ────────────────────────────────────────────────────────────────
    # Dataset
    # ────────────────────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )

    print("Formatting dataset for SFT...")
    dataset = dataset.map(format_example, batched=False, remove_columns=dataset["train"].column_names)

    # ────────────────────────────────────────────────────────────────
    # Training Arguments with W&B enabled
    # ────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",                     # ← KEY CHANGE: enable W&B logging
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    # ────────────────────────────────────────────────────────────────
    # Trainer
    # ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer
    )

    print("Starting SFT training with W&B tracking...")
    trainer.train()

    # Save final adapter
    final_save_path = Path(OUTPUT_DIR) / "final_adapter"
    trainer.save_model(final_save_path)
    print(f"Training complete. Adapter saved to: {final_save_path}")

    # Finish W&B run (good practice)
    wandb.finish()


if __name__ == "__main__":
    main()