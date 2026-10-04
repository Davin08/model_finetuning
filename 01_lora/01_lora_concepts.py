"""
01_lora_concepts.py — Phase 1: See LoRA adapters in action (no training needed)

This script shows you the MATH behind LoRA using small example matrices.
Run: python 01_lora_concepts.py
"""

import torch
import torch.nn as nn

def demo_rank_decomposition():
    """Show how a large weight update can be approximated by two small matrices."""
    print("\n" + "─" * 65)
    print("📐 DEMO 1: Low-Rank Matrix Decomposition")
    print("─" * 65)
    print("""
LoRA's core idea: A weight matrix update ΔW can be approximated
by the product of two small matrices A and B.

  ΔW  ≈  A × B
  
  Shape of ΔW: (d_in, d_out)  — e.g. (4096, 4096) = 16M params
  Shape of A:  (d_in, r)      — e.g. (4096, 8)    = 32K params
  Shape of B:  (r, d_out)     — e.g. (8, 4096)    = 32K params

With r=8, we represent a 16M-param update with only 64K params!
That's 0.4% of the original size.
""")

    # Small example to show the math
    d_in, d_out, r = 6, 6, 2

    # Simulate what a "true" weight update might look like
    torch.manual_seed(42)
    true_delta_W = torch.randn(d_in, d_out) * 0.1

    # LoRA approximation: A @ B
    A = torch.randn(d_in, r) * 0.1   # initialized randomly
    B = torch.zeros(r, d_out)          # initialized to ZERO (important!)
    lora_delta_W = A @ B

    print(f"Example shapes (d_in={d_in}, d_out={d_out}, r={r}):")
    print(f"  True ΔW:   {true_delta_W.shape}  ({d_in * d_out} params)")
    print(f"  A matrix:  {A.shape}  ({d_in * r} params)")
    print(f"  B matrix:  {B.shape}  ({r * d_out} params)")
    print(f"  A @ B:     {lora_delta_W.shape}  (same shape, {d_in * r + r * d_out} total params)")
    print(f"\n  Compression: {d_in * d_out} → {d_in * r + r * d_out} params ({100*(d_in*r+r*d_out)/(d_in*d_out):.1f}%)")
    print("\n  Note: B starts at zero → LoRA starts as identity (no change)")
    print("        As training proceeds, A and B learn the update.")


def demo_lora_forward_pass():
    """Show the LoRA forward pass formula: output = W_original(x) + (A @ B)(x)."""
    print("\n" + "─" * 65)
    print("⚡ DEMO 2: LoRA Forward Pass")
    print("─" * 65)
    print("""
During fine-tuning, the forward pass is:

  output = W_original @ x  +  scaling * (B @ A) @ x
                           ↑
                     LoRA contribution (starts at 0)

  W_original: FROZEN (no gradients, no updates, stays the same)
  A, B:       TRAINED (get updated every step)
  scaling:    α / r  (controls how much LoRA contributes)
""")

    d, r = 8, 2
    batch_size = 3

    W_original = torch.randn(d, d)          # Frozen pre-trained weights
    A = torch.randn(d, r) * 0.1            # LoRA A (trained)
    B = torch.zeros(r, d)                   # LoRA B (starts at zero)
    alpha = 16.0
    scaling = alpha / r                      # = 8.0

    x = torch.randn(batch_size, d)           # Input

    # Standard forward pass (no LoRA)
    out_original = x @ W_original.T

    # LoRA forward pass
    lora_update = x @ A @ B * scaling       # LoRA contribution
    out_lora = out_original + lora_update    # Combined

    print(f"Input x: {x.shape}")
    print(f"W_original: {W_original.shape} (frozen)")
    print(f"A: {A.shape}, B: {B.shape} (trainable)")
    print(f"Scaling (α/r): {scaling}")
    print(f"\nOriginal output: {out_original.shape}")
    print(f"LoRA update (B=0): {lora_update.abs().sum():.4f} (zero at start!)")
    print(f"Combined output: {out_lora.shape}")
    print("\n✅ At the start of training, LoRA adds nothing (B=0).")
    print("   As B learns, it contributes more and more to the output.")


def demo_lora_vs_full_params():
    """Compare parameter counts: full fine-tuning vs LoRA."""
    print("\n" + "─" * 65)
    print("📊 DEMO 3: Parameter Count Comparison")
    print("─" * 65)

    models = [
        ("GPT-2 Small",  768,   12,   12),   # d_model, n_layers, n_heads
        ("Phi-2",       2560,   32,   32),
        ("Mistral 7B",  4096,   32,   32),
        ("Llama-3 8B",  4096,   32,   32),
    ]

    for name, d, n_layers, n_heads in models:
        # Attention Q, K, V, O projections per layer
        attn_params = 4 * d * d * n_layers

        # LoRA with r=8, only on Q and V
        r = 8
        lora_params = 2 * (d * r + r * d) * n_layers  # A + B for Q and V

        pct = 100 * lora_params / attn_params

        print(f"\n  {name}:")
        print(f"    Full attention params: {attn_params/1e6:.1f}M")
        print(f"    LoRA params (r=8, Q+V): {lora_params/1e3:.0f}K  ({pct:.2f}%)")
        print(f"    Savings: {(1-pct/100)*100:.1f}% fewer parameters to train!")


def demo_lora_in_transformers():
    """Show how peft library wraps a model with LoRA."""
    print("\n" + "─" * 65)
    print("🤗 DEMO 4: LoRA in HuggingFace PEFT Library")
    print("─" * 65)
    print("""
In code, you use the `peft` library to add LoRA to any HuggingFace model:

```python
from peft import LoraConfig, get_peft_model, TaskType

# 1. Define LoRA configuration
lora_config = LoraConfig(
    r=8,                          # Rank — start here, increase if needed
    lora_alpha=16,                # Scaling factor (usually 2x rank)
    lora_dropout=0.05,            # Regularization
    bias="none",                  # Don't add LoRA to bias terms
    task_type=TaskType.CAUSAL_LM, # We're doing language modeling
    target_modules=[              # Which layers to apply LoRA to
        "q_proj",                 # Query projection (attention)
        "v_proj",                 # Value projection (attention)
    ],
)

# 2. Wrap model with LoRA
model = get_peft_model(base_model, lora_config)

# 3. Check trainable params
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 2,779,683,840 || trainable%: 0.15%

# 4. Train exactly like a normal PyTorch model
trainer.train()

# 5. Save only the small adapter (not the full model!)
model.save_pretrained("./my_payroll_adapter")  # ~50 MB instead of 5 GB
```
""")

    print("Key insight: You share the base model (5 GB) across many adapters.")
    print("Each adapter (50 MB) specializes for a different task.")
    print("This is perfect for running multiple fine-tuned models simultaneously!")


def main():
    print("=" * 65)
    print("  Phase 1: LoRA Concepts Demo")
    print("  Understanding Low-Rank Adaptation")
    print("=" * 65)

    demo_rank_decomposition()
    demo_lora_forward_pass()
    demo_lora_vs_full_params()
    demo_lora_in_transformers()

    print("\n" + "=" * 65)
    print("🎉 LoRA concepts demo complete!")
    print("""
Key takeaways:
  1. LoRA adds small A×B matrices alongside frozen original weights
  2. B starts at zero → no impact at start, grows during training
  3. Typical LoRA trains only 0.1–1% of parameters
  4. Adapter files are tiny (~50 MB) and shareable
  5. Multiple adapters can be stacked on the same base model

👉 Next: python 02_generate_synthetic_data.py
""")
    print("=" * 65)


if __name__ == "__main__":
    main()
