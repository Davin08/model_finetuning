# Phase 0: Foundations & Environment Setup

## 🎯 Learning Objectives
By the end of this phase you will understand:
1. What LLMs are and how transformers work (at the intuition level)
2. The difference between pre-training, fine-tuning, and prompting
3. Key vocabulary: tokens, embeddings, attention, parameters, gradients
4. Your Apple Silicon Mac setup for training (MPS backend)

---

## 📖 Concept 1: What is an LLM?

A **Large Language Model (LLM)** is a neural network trained to predict the next token in a sequence.

```
Input:  "The employee's gross salary is"
Output: " $4,200.00"   ← the model predicts this token
```

That's it at its core. The "magic" comes from training on trillions of tokens — the model learns grammar, facts, reasoning, and structure implicitly.

### Key numbers to know:
| Model Size | Approx. Parameters | RAM needed (fp16) |
|------------|-------------------|-------------------|
| Small (GPT-2) | 125M | ~0.25 GB |
| Medium (Phi-2) | 2.7B | ~5.4 GB |
| Large (Mistral-7B) | 7B | ~14 GB |
| XL (Llama-3 70B) | 70B | ~140 GB |

Your M-series Mac can comfortably run 2B–7B models.

---

## 📖 Concept 2: Pre-training vs Fine-tuning vs Prompting

```
┌─────────────────────────────────────────────────────────────┐
│  PRE-TRAINING                                               │
│  Train on trillions of tokens from the internet             │
│  Cost: $1M–$10M+   Time: months   Done by: OpenAI, Meta... │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  FINE-TUNING  ← THIS IS WHAT WE'RE DOING                   │
│  Continue training on YOUR specific data (1K–100K samples)  │
│  Cost: $0–$50     Time: minutes–hours   Done by: YOU        │
│  Teaches model domain knowledge + output format             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  PROMPTING / RAG                                            │
│  No training — just clever instructions at inference time   │
│  Cost: near-zero   Time: instant   No code needed           │
└─────────────────────────────────────────────────────────────┘
```

### When to fine-tune vs just prompt?

| Situation | Solution |
|-----------|----------|
| Model doesn't know your domain terminology | Fine-tune |
| You want a specific output format (JSON, tables) | Fine-tune |
| You need consistent behavior across 1000s of runs | Fine-tune |
| You have a few examples and just testing | Prompt engineering |
| Your data is too sensitive to send to an API | Fine-tune (local) |

**Your case:** Payroll PDFs → Fine-tune is perfect! You need structured JSON output with specific fields.

---

## 📖 Concept 3: Transformers in Plain English

Transformers are the architecture behind all modern LLMs. Here's the intuition:

### Tokenization
Text is split into chunks called **tokens** (roughly 3/4 of a word):
```
"payroll" → ["pay", "roll"]
"$4,200.00" → ["$", "4", ",", "200", ".", "00"]
```

### Embeddings
Each token is converted to a list of numbers (a vector) that captures meaning:
```
"salary" → [0.23, -0.81, 0.44, ..., 0.12]  (768 or 4096 numbers)
"wage"   → [0.21, -0.79, 0.46, ..., 0.11]  (similar! they're close)
```

### Attention
The model learns which tokens to "pay attention to" when predicting the next one:
```
"The gross pay shown on the stub is ___"
                                    ↑
  The model attends to "gross pay" and "stub" to predict the next token
```

### That's the core loop:
1. Tokenize input
2. Embed tokens → vectors
3. Pass through N "transformer layers" (attention + feedforward)
4. Output probability over vocabulary → pick next token

---

## 📖 Concept 4: What Fine-Tuning Actually Does

During fine-tuning, we run the **same training loop** as pre-training, just with:
- **Your data** instead of the internet
- **Fewer steps** (100s–1000s instead of billions)
- **Lower learning rate** (so we don't forget what the model already knows)

### The training loop:
```python
for batch in your_dataset:
    # 1. Forward pass: model makes a prediction
    output = model(input_tokens)
    
    # 2. Loss: how wrong was it?
    loss = cross_entropy(output, target_tokens)
    
    # 3. Backward pass: compute gradients (which direction to nudge weights)
    loss.backward()
    
    # 4. Optimizer step: nudge weights slightly in the right direction
    optimizer.step()
```

This repeats thousands of times until the model gets good at your task.

---

## 🛠️ Setup: Your Apple Silicon Environment

### Step 1: Check your Mac specs
```bash
# Run this in terminal
system_profiler SPHardwareDataType | grep -E "Chip|Memory"
```

### Step 2: Install dependencies
Run the setup script:
```bash
cd 00_foundations
python setup_check.py
```

### Step 3: Verify MPS (Metal Performance Shaders) works
```bash
python verify_mps.py
```

You should see: `✅ MPS is available! Training will use your GPU cores.`

---

## 🏋️ Exercise

After running setup, open `01_tokenization_demo.py` and run it:
```bash
python 01_tokenization_demo.py
```

This will show you how a real tokenizer breaks down payroll text into tokens — the exact first step in every LLM pipeline.

---

## ✅ Phase 0 Checklist

- [ ] Read all 4 concepts above
- [ ] Run `setup_check.py` successfully
- [ ] Run `verify_mps.py` — see MPS available
- [ ] Run `01_tokenization_demo.py` — understand token output
- [ ] Move on to `../01_lora/`
