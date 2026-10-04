"""
03_train_lora.py — Phase 1: Fine-tune a model with LoRA

We fine-tune microsoft/phi-2 (2.7B params) with LoRA on our payroll data.
Phi-2 is a great choice for Apple Silicon — it fits comfortably on M-series Macs.

Run: python 03_train_lora.py
Expected time: 15-30 minutes on M2/M3 Mac
"""

import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── Configuration ─────────────────────────────────────────────────────────────
# We'll use a small model that fits well on Apple Silicon Macs.
# Phi-2 is excellent: 2.7B params, strong reasoning, MIT licensed.
MODEL_NAME = "microsoft/phi-2"

# LoRA hyperparameters (good defaults for a classification/extraction task)
LORA_CONFIG = {
    "r": 8,                 # Rank — controls capacity of adapter
    "lora_alpha": 16,       # Scaling: use 2x rank
    "lora_dropout": 0.05,   # Regularization
    "target_modules": [     # Attention layers to adapt
        "q_proj",
        "v_proj",
        "k_proj",
        "dense",            # Phi-2 uses "dense" instead of "o_proj"
    ],
}

# Training hyperparameters
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 1,         # Keep at 1 for Mac (MPS memory constraints)
    "grad_accum_steps": 8,   # Effective batch = 1 * 8 = 8
    "learning_rate": 2e-4,
    "max_seq_length": 512,   # Max tokens per sample
    "warmup_ratio": 0.05,
    "output_dir": "./outputs/lora_phi2_payroll",
    "logging_steps": 10,
    "save_steps": 50,
    "eval_steps": 50,
}

# ── Device setup ──────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
        print("🖥️  Using Apple Silicon MPS (GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🖥️  Using NVIDIA CUDA GPU")
    else:
        device = "cpu"
        print("🖥️  Using CPU (training will be slow)")
    return device


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(train_path: str = "data/train.jsonl", eval_path: str = "data/eval.jsonl"):
    """Load JSONL data and format for training."""
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_raw = load_jsonl(train_path)
    eval_raw = load_jsonl(eval_path)

    def format_prompt(sample):
        """Format as Phi-2 instruction format."""
        return {
            "text": (
                f"Instruct: {sample['instruction']}\n\n"
                f"Document:\n{sample['input']}\n\n"
                f"Output: {sample['output']}"
            )
        }

    train_dataset = Dataset.from_list([format_prompt(s) for s in train_raw])
    eval_dataset  = Dataset.from_list([format_prompt(s) for s in eval_raw])

    print(f"✅ Loaded {len(train_dataset)} training samples")
    print(f"✅ Loaded {len(eval_dataset)} eval samples")
    print(f"\nSample prompt (first 300 chars):\n{train_dataset[0]['text'][:300]}...\n")

    return train_dataset, eval_dataset


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, device: str):
    """Load base model and tokenizer."""
    print(f"\n⏳ Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Phi-2 doesn't have a pad token by default — add one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"⏳ Loading model: {model_name}")
    print("   (This downloads ~5 GB on first run — subsequent runs use cache)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        trust_remote_code=True,
        # Note: on MPS we don't use device_map="auto" — we'll move manually
    )

    # Move to MPS / CUDA if available
    if device in ["mps", "cuda"]:
        model = model.to(device)

    print(f"✅ Model loaded on {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params / 1e9:.2f}B")

    return model, tokenizer


# ── LoRA setup ────────────────────────────────────────────────────────────────
def apply_lora(model):
    """Wrap model with LoRA adapters."""
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable vs total params
    model.print_trainable_parameters()

    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, tokenizer, train_dataset, eval_dataset):
    """Run the fine-tuning loop."""
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)

    training_args = SFTConfig(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["grad_accum_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=False,           # MPS doesn't support fp16 training ops (use bf16 or fp32)
        bf16=False,           # Apple Silicon doesn't support bf16 in older PyTorch
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",     # Set to "wandb" if you have wandb installed
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        # MPS-specific: disable flash attention (not supported on MPS)
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\n🚀 Starting LoRA fine-tuning!")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['grad_accum_steps']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Output: {TRAINING_CONFIG['output_dir']}")
    print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("\n   Watch the training loss go DOWN — that's learning happening! 📉\n")

    trainer.train()

    print(f"\n✅ Training complete! Finished at: {datetime.now().strftime('%H:%M:%S')}")
    return trainer


# ── Save ──────────────────────────────────────────────────────────────────────
def save_adapter(model, tokenizer):
    """Save only the LoRA adapter weights (tiny!) not the full model."""
    adapter_path = f"{TRAINING_CONFIG['output_dir']}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Check size
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(adapter_path)
        for f in files
    )
    print(f"\n✅ Adapter saved to: {adapter_path}")
    print(f"   Adapter size: {total_size / 1e6:.1f} MB  (vs ~5000 MB for full model)")
    print("\n   Share this adapter with anyone who has the same base model!")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 1: LoRA Fine-Tuning — Payroll Extraction")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 65)

    # 1. Check data exists
    if not os.path.exists("data/train.jsonl"):
        print("❌ Training data not found. Run first: python 02_generate_synthetic_data.py")
        return

    # 2. Get device
    device = get_device()

    # 3. Load data
    print("\n📦 Loading training data...")
    train_dataset, eval_dataset = load_data()

    # 4. Load model
    print("\n🤖 Loading base model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)

    # 5. Apply LoRA
    print("\n🔧 Applying LoRA adapters...")
    model = apply_lora(model)

    # 6. Train
    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    # 7. Save
    print("\n💾 Saving LoRA adapter...")
    save_adapter(model, tokenizer)

    print("\n" + "=" * 65)
    print("🎉 Fine-tuning complete!")
    print("""
Your model has been fine-tuned to extract payroll data!

Next steps:
  👉 python 04_inference.py    — test your model on new payroll text
  👉 python 05_evaluate.py     — measure accuracy metrics
""")
    print("=" * 65)


if __name__ == "__main__":
    main()
