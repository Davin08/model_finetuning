# Phase 1: LoRA Fine-Tuning

> **Prerequisite:** Complete `../00_foundations/` first.

## 🎯 Learning Objectives
By the end of this phase you will:
1. Understand *why* LoRA works (the math intuition)
2. Fine-tune a real model using `peft` + `trl`
3. Have a trained model that extracts fields from payroll text

---

## 📖 Concept: What is LoRA?

**LoRA = Low-Rank Adaptation**

Full fine-tuning updates ALL weights of a model:
- **Phi-2 (2.7B params):** 2.7 **billion** weights to update → ~5.4 GB gradients + ~10.8 GB optimizer states → **~22 GB VRAM required!**
- That exceeds what consumer laptops can train efficiently.

LoRA's insight: **Weight updates have a low "intrinsic dimension"**. We freeze the giant base model and inject two tiny low-rank adapter matrices ($A$ and $B$) alongside each attention projection:

```
FULL FINE-TUNING:
  W_new = W_original + ΔW              ← ΔW has the same massive shape as W_original

LORA:
  W_new = W_original + (A × B) * (α/r) ← A and B are ultra-compact bottleneck matrices!
```

---

### 📐 How Tensor Shapes Vary Across Models

The shape of the weight matrices depends directly on the model's hidden dimension (`d_model`):

| Model | Parameters | Layers | Attention Heads | `d_model` ($D$) | Attention Matrix Shape ($W$) | LoRA $A$ ($D \times r$) | LoRA $B$ ($r \times D$) | Trainable % ($r=8$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2 Small** | 125M | 12 | 12 | **768** | `(768, 768)` | `(768, 8)` | `(8, 768)` | ~1.04% |
| **Phi-2 (Ours)** ⭐ | 2.7B | 32 | 32 | **2,560** | `(2560, 2560)` | `(2560, 8)` | `(8, 2560)` | **~0.31%** |
| **Mistral 7B** | 7B | 32 | 32 | **4,096** | `(4096, 4096)` | `(4096, 8)` | `(8, 4096)` | ~0.20% |
| **Llama-3 8B** | 8B | 32 | 32 | **4,096** | `(4096, 4096)` | `(4096, 8)` | `(8, 4096)` | ~0.20% |
| **Llama-3 70B** | 70B | 80 | 64 | **8,192** | `(8192, 8192)` | `(8192, 8)` | `(8, 8192)` | ~0.02% |

> Notice: As the base model grows from 2.7B to 70B, **LoRA becomes even more efficient** because the rank $r=8$ stays tiny!

---

### 🧊 3D Isometric View: The Low-Rank Matrix Bottleneck

```
                +------------------------------------+
               /                                    /|
              /         FROZEN BASE WEIGHT         / |
             /             TENSOR CUBE            /  |
            +------------------------------------+   |
            |                                    |   |   d_model
            |             W_original             |   |   (Phi-2: 2560)
            |           (2560 x 2560)            |   +   (Llama: 4096)
            |                                    |  /
            |          6,553,600 params          | /
            |         ❄️ [LOCKED / NO GRADS]     |/  d_model (2560)
            +------------------------------------+
                               +
              +--------+              +------------------------------------+
             /        /|             /                                    /|
            +--------+ |            +------------------------------------+ |
            |   A    | |  r=8       |                 B                  | |  r=8
            | (Down) | | (Bottleneck|             (Up-Proj)              | +
            +--------+/   Prism)    +------------------------------------+/  d_model
            d_model (2560)                       (8 x 2560)                  (2560)
             20,480 params                      20,480 params
             🔥 [TRAINED]                       🔥 [TRAINED, starts @ 0]
```

---

## 🎨 3D Architecture Diagrams

### 1️⃣ 3D Tensor Flow: Input Volume through the 32-Layer Stack

```mermaid
flowchart TD
    subgraph TENSOR_IN["🧊 3D Input Activation Tensor: [Batch × Seq_Len × d_model]"]
        direction TB
        IN_3D["📦 Input Batch Volume: <b>[1 × 512 × 2560]</b><br/>Tokens converted to 2560-dimensional embeddings"]
    end

    subgraph STACK_3D["📚 3D Transformer Stack (Phi-2: 32 Layers in Depth)"]
        direction TB
        L32["┌──────────────────────────────────────────────────┐<br/>│ 🔷 Layer 32: Top Block [Residual + Attention + MLP] │<br/>└──────────────────────────────────────────────────┘"]
        L_DOTS["  ⋮  ⋮  ⋮   [Layers 3 to 31 Stacked Vertically in Depth]   ⋮  ⋮  ⋮  "]
        L2["┌──────────────────────────────────────────────────┐<br/>│ 🔷 Layer 2: Hidden Block [Attention + MLP Layers]  │<br/>└──────────────────────────────────────────────────┘"]
        L1["┌──────────────────────────────────────────────────┐<br/>│ 🔷 Layer 1: Base Block [Attention + MLP Layers]    │<br/>└──────────────────────────────────────────────────┘"]
        L1 --> L2 --> L_DOTS --> L32
    end

    subgraph TENSOR_OUT["🧊 3D Output Probability Volume"]
        OUT_3D["📦 Next-Token Logits: <b>[1 × 512 × 51,200]</b><br/>Softmax produces probabilities across 51.2K vocab"]
    end

    IN_3D ==> STACK_3D
    STACK_3D ==> TENSOR_OUT

    style IN_3D fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style L1 fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    style L2 fill:#1e1b4b,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    style L_DOTS fill:#020617,stroke:#475569,stroke-dasharray: 5 5,color:#94a3b8
    style L32 fill:#311042,stroke:#ec4899,stroke-width:2px,color:#f8fafc
    style OUT_3D fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc
```

---

### 2️⃣ 3D Multi-Head Attention: Slicing the Hidden Dimension

*Phi-2 has $D=2560$ split across $H=32$ heads. Each head operates on a 3D slice of dimension $d_k = 80$:*

```mermaid
flowchart LR
    X_IN["📦 Layer Input X<br/><b>[1 × Seq × 2560]</b>"]

    subgraph PROJECTIONS["3D Weight Cylinders: [d_model × d_model]"]
        direction TB
        W_Q[("🛢️ W_Query Tensor<br/>(2560 × 2560)")]
        W_K[("🛢️ W_Key Tensor<br/>(2560 × 2560)")]
        W_V[("🛢️ W_Value Tensor<br/>(2560 × 2560)")]
    end

    subgraph HEADS_3D["🧩 3D Multi-Head Slices: [32 Heads × Seq_Len × 80]"]
        direction TB
        H1["Head 01 Slice [T × 80]"]
        H2["Head 02 Slice [T × 80]"]
        H_MORE["⋮  [32 Parallel Attention Slices]  ⋮"]
        H32["Head 32 Slice [T × 80]"]
    end

    subgraph ATTN_OUT["⚡ Attention Volume & Output Projection"]
        direction TB
        MAT["Scores = Softmax(Q·Kᵀ / √80) · V"]
        W_O[("🛢️ W_Output Projection<br/>(2560 × 2560)")]
        MAT --> W_O
    end

    X_IN ==> W_Q & W_K & W_V
    W_Q & W_K & W_V ==> HEADS_3D
    HEADS_3D ==> ATTN_OUT
    ATTN_OUT ==> Y_OUT["📦 Layer Output<br/><b>[1 × Seq × 2560]</b>"]

    style X_IN fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style W_Q fill:#1e3a5f,stroke:#38bdf8,stroke-width:2px,color:#bae6fd
    style W_K fill:#1e3a5f,stroke:#38bdf8,stroke-width:2px,color:#bae6fd
    style W_V fill:#1e3a5f,stroke:#38bdf8,stroke-width:2px,color:#bae6fd
    style W_O fill:#1e3a5f,stroke:#38bdf8,stroke-width:2px,color:#bae6fd
    style H1 fill:#1e1b4b,stroke:#a855f7,stroke-width:1px,color:#f3e8ff
    style H2 fill:#1e1b4b,stroke:#a855f7,stroke-width:1px,color:#f3e8ff
    style H32 fill:#1e1b4b,stroke:#a855f7,stroke-width:1px,color:#f3e8ff
    style Y_OUT fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#a7f3d0
```

---

### 3️⃣ 3D LoRA Adapter Injection: Frozen Base vs. Trainable Bottleneck

*Gradients bypass the 6.55M parameter frozen base cylinder and only update the 40.9K adapter parameters:*

```mermaid
flowchart TD
    X["📦 Activation Vector x: [1 × 2560]"]

    subgraph FROZEN_PATH["❄️ FROZEN BASE PATH (No Gradients Computed)"]
        direction TB
        W0[("🧊 Base Weight Cylinder: W_orig<br/><b>[2560 × 2560]</b> (6,553,600 weights)<br/>🔒 requires_grad = False")]
        W0_OUT["Base Projection: x · W_orig<br/>Shape: [1 × 2560]"]
        W0 --> W0_OUT
    end

    subgraph LORA_PATH["🔥 TRAINABLE LoRA BOTTLENECK (Requires Grad)"]
        direction LR
        subgraph DOWN_PROJ["Down-Projection Prism"]
            A["📐 Matrix A: [2560 × 8]<br/>(20,480 weights)<br/><i>Shrinks 2560 → 8</i>"]
        end
        subgraph BOTTLENECK["Rank Core"]
            R["⚡ Rank Dimension<br/>r = 8"]
        end
        subgraph UP_PROJ["Up-Projection Prism"]
            B["📐 Matrix B: [8 × 2560]<br/>(20,480 weights)<br/><i>Expands 8 → 2560</i>"]
        end
        A --> R --> B
    end

    SCALE["⚖️ Scaling Multiplier: (α / r = 16 / 8 = 2.0)"]
    SUM["➕ Elementwise Sum: h = x·W_orig + (x·A·B) * (α/r)"]
    OUT["📦 Final Projection Vector: [1 × 2560]"]

    X ==> W0
    X ==> A
    B ==> SCALE
    W0_OUT ==> SUM
    SCALE ==> SUM
    SUM ==> OUT

    style W0 fill:#111827,stroke:#ef4444,stroke-width:3px,color:#f87171
    style A fill:#0c4a6e,stroke:#38bdf8,stroke-width:2px,color:#bae6fd
    style R fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#e9d5ff
    style B fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#a7f3d0
    style SUM fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#fde68a
    style X fill:#0f172a,stroke:#94a3b8,stroke-width:2px,color:#f1f5f9
    style OUT fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#bbf7d0
```

---

### 4️⃣ 3D Memory & Precision Architecture: Full vs. LoRA vs. QLoRA

```mermaid
flowchart LR
    subgraph FULL["1️⃣ Full Fine-Tuning (FP16/FP32)"]
        direction TB
        F_W[("🛢️ Weights [FP16]<br/>5.4 GB")]
        F_G[("🛢️ Gradients [FP16]<br/>5.4 GB")]
        F_O[("🛢️ Optimizer States [FP32]<br/>10.8 GB")]
        F_TOT["💥 Total VRAM: ~22 GB+<br/><b>Exceeds base Mac memory!</b>"]
        F_W --- F_G --- F_O --- F_TOT
    end

    subgraph LORA["2️⃣ LoRA (Phase 1: Our Current Setup)"]
        direction TB
        L_W[("🧊 Frozen Base [FP16]<br/>5.4 GB (No Gradients)")]
        L_A[("🔥 Adapters A & B [FP16]<br/>~50 MB Trainable")]
        L_O[("⚙️ Optimizer States<br/>~100 MB")]
        L_TOT["✅ Total VRAM: ~6 GB<br/><b>Runs smoothly on Apple Silicon!</b>"]
        L_W --- L_A --- L_O --- L_TOT
    end

    subgraph QLORA["3️⃣ QLoRA (Phase 2: 4-Bit NF4)"]
        direction TB
        Q_W[("🧊 Quantized Base [NF4 4-bit]<br/>~1.6 GB (Compressed 4x!)")]
        Q_A[("🔥 Adapters A & B [FP16]<br/>~50 MB Trainable")]
        Q_O[("⚙️ Optimizer States<br/>~100 MB")]
        Q_TOT["🚀 Total VRAM: ~2.5 GB<br/><b>Fits 7B+ models on 8GB Mac!</b>"]
        Q_W --- Q_A --- Q_O --- Q_TOT
    end

    style FULL fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fee2e2
    style LORA fill:#082f49,stroke:#0ea5e9,stroke-width:2px,color:#e0f2fe
    style QLORA fill:#1e1b4b,stroke:#a855f7,stroke-width:2px,color:#f3e8ff
```

---

### The rank (r) hyperparameter:
| r value | Trainable params | Use case |
|---------|-----------------|----------|
| r=4 | Very few | Simple classification, limited training data |
| r=8 | Few (default) | Standard instruction tuning & extraction (our project) |
| r=16 | Moderate | Complex domain adaptation (medical, legal) |
| r=64 | Many | Near full fine-tuning capacity |

### Which layers to apply LoRA to?
```python
target_modules = ["q_proj", "v_proj"]  # Attention query and value projections
# For Phi-2, we also adapt "k_proj" and "dense" for maximum extraction fidelity!
```

---

## 📖 Concept: Instruction Fine-Tuning Format

We're doing **Supervised Fine-Tuning (SFT)** — teaching the model to follow instructions.

Your data must be in **instruction format**:
```json
{
  "instruction": "Extract the following fields from this payroll stub as JSON: employee_name, gross_pay, net_pay, pay_period",
  "input": "ACME Corp - Pay Stub\nEmployee: Maria Garcia\nPeriod: Oct 1-15 2024\nGross: $3,800.00\nFederal Tax: -$570.00\nNet Pay: $3,056.25",
  "output": "{\"employee_name\": \"Maria Garcia\", \"gross_pay\": 3800.00, \"net_pay\": 3056.25, \"pay_period\": \"Oct 1-15 2024\"}"
}
```

The model is trained to look at `instruction + input` and produce `output`.

---

## 🗂️ Files in This Phase

| File | What it teaches |
|------|----------------|
| `01_lora_concepts.py` | Interactive LoRA math demo — see the adapter matrices |
| `02_generate_synthetic_data.py` | Create payroll training data (no real PDFs needed yet!) |
| `03_train_lora.py` | **Main training script** — fine-tune Phi-2 with LoRA |
| `04_inference.py` | Load your trained adapter and test it |
| `05_evaluate.py` | Measure your model's accuracy |
| `data/` | Training and eval datasets |
| `outputs/` | Saved model checkpoints |

---

## 🚀 Quick Start

```bash
# Step 1: Understand LoRA adapters visually
python 01_lora_concepts.py

# Step 2: Generate synthetic payroll training data
python 02_generate_synthetic_data.py

# Step 3: Train! (takes 10-30 min on Apple Silicon M-series)
python 03_train_lora.py

# Step 4: Test your model
python 04_inference.py

# Step 5: Measure accuracy
python 05_evaluate.py
```

---

## 📊 Expected Results

After training on ~200 synthetic payroll samples for 3 epochs:
- **Exact field extraction accuracy:** ~85-92%
- **JSON parsing success rate:** ~95%
- **Training time on M2:** ~15-20 minutes
- **Adapter size on disk:** ~50 MB (vs 5.4 GB for the full model!)

---

## ✅ Phase 1 Checklist

- [ ] Read and understand the LoRA concept above
- [ ] Run `01_lora_concepts.py`
- [ ] Run `02_generate_synthetic_data.py` → inspect `data/train.jsonl`
- [ ] Run `03_train_lora.py` → watch loss go down!
- [ ] Run `04_inference.py` → test your model on new payroll text
- [ ] Run `05_evaluate.py` → see accuracy metrics
- [ ] Move on to `../02_qlora/`
