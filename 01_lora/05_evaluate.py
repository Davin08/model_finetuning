"""
05_evaluate.py — Phase 1: Measure your model's extraction accuracy

Computes metrics on the eval set to understand model quality.
Run: python 05_evaluate.py
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "./outputs/lora_phi2_payroll/final_adapter"
EVAL_PATH = "data/eval.jsonl"

INSTRUCTION = (
    "You are a payroll document parser. Extract all relevant financial fields from "
    "the following payroll document text and return them as a valid JSON object. "
    "Include document_type, employee information, all monetary amounts, and dates. "
    "Return ONLY the JSON, no explanation."
)

NUMERIC_FIELDS = ["gross_pay", "net_pay", "federal_tax", "state_tax",
                  "social_security", "medicare", "total_deductions"]
STRING_FIELDS  = ["document_type", "employee_name", "pay_frequency"]


def load_eval_data(path: str = EVAL_PATH, max_samples: int = 30):
    """Load eval samples (limit to 30 for speed)."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break
    return samples


def load_model():
    """Load base model + LoRA adapter."""
    import os
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter not found at {ADAPTER_PATH}")
        print("   Run training first: python 03_train_lora.py")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, document_text: str):
    """Run extraction on one document."""
    prompt = (
        f"Instruct: {INSTRUCTION}\n\n"
        f"Document:\n{document_text.strip()}\n\n"
        f"Output:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(text: str):
    """Extract JSON from model output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def compute_metrics(predictions, ground_truths):
    """Compute accuracy metrics."""
    n = len(predictions)
    json_parse_success = 0
    field_exact_match = {f: 0 for f in STRING_FIELDS}
    field_within_1pct = {f: 0 for f in NUMERIC_FIELDS}
    field_total = {f: 0 for f in STRING_FIELDS + NUMERIC_FIELDS}

    for pred_raw, gt_dict in zip(predictions, ground_truths):
        pred = parse_json(pred_raw)
        if pred is None:
            continue
        json_parse_success += 1

        # String fields — exact match (case-insensitive)
        for field in STRING_FIELDS:
            if field in gt_dict:
                field_total[field] += 1
                pred_val = str(pred.get(field, "")).lower().strip()
                gt_val = str(gt_dict[field]).lower().strip()
                if pred_val == gt_val:
                    field_exact_match[field] += 1

        # Numeric fields — within 1% tolerance
        for field in NUMERIC_FIELDS:
            if field in gt_dict and gt_dict[field] != 0:
                field_total[field] += 1
                try:
                    pred_val = float(pred.get(field, -1))
                    gt_val = float(gt_dict[field])
                    if abs(pred_val - gt_val) / gt_val <= 0.01:  # within 1%
                        field_within_1pct[field] += 1
                except (ValueError, TypeError):
                    pass

    results = {
        "total_samples": n,
        "json_parse_rate": json_parse_success / n if n > 0 else 0,
        "string_field_accuracy": {
            f: (field_exact_match[f] / field_total[f]) if field_total[f] > 0 else 0
            for f in STRING_FIELDS
        },
        "numeric_field_accuracy_1pct": {
            f: (field_within_1pct[f] / field_total[f]) if field_total[f] > 0 else 0
            for f in NUMERIC_FIELDS
        },
    }

    # Overall accuracy = average across all fields
    all_accs = list(results["string_field_accuracy"].values()) + \
               list(results["numeric_field_accuracy_1pct"].values())
    results["overall_field_accuracy"] = sum(all_accs) / len(all_accs) if all_accs else 0

    return results


def print_metrics(metrics):
    """Pretty-print evaluation results."""
    print("\n" + "─" * 60)
    print("📊 EVALUATION RESULTS")
    print("─" * 60)
    print(f"  Samples evaluated:      {metrics['total_samples']}")
    print(f"  JSON parse success:     {metrics['json_parse_rate']:.1%}")
    print(f"  Overall field accuracy: {metrics['overall_field_accuracy']:.1%}")

    print("\n  String Fields (exact match):")
    for field, acc in metrics["string_field_accuracy"].items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {field:25} [{bar}] {acc:.1%}")

    print("\n  Numeric Fields (within 1% tolerance):")
    for field, acc in metrics["numeric_field_accuracy_1pct"].items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {field:25} [{bar}] {acc:.1%}")

    print("\n" + "─" * 60)
    overall = metrics["overall_field_accuracy"]
    if overall >= 0.90:
        print("🏆 Excellent! Your model is production-ready.")
    elif overall >= 0.75:
        print("✅ Good! Consider more training data or epochs to improve.")
    elif overall >= 0.50:
        print("⚠️  Moderate. Try: more epochs, higher rank, more data.")
    else:
        print("❌ Low accuracy. Check data format and training loss.")


def main():
    print("=" * 65)
    print("  Phase 1: Model Evaluation — Payroll Extraction")
    print("=" * 65)

    import os
    if not os.path.exists(EVAL_PATH):
        print("❌ Eval data not found. Run: python 02_generate_synthetic_data.py")
        return

    model, tokenizer = load_model()
    if model is None:
        return

    print(f"\n📦 Loading eval data from {EVAL_PATH}...")
    eval_samples = load_eval_data(max_samples=30)
    print(f"✅ Loaded {len(eval_samples)} eval samples")

    print("\n🤖 Running inference on eval set...")
    print("   (This takes ~1-2 min on Apple Silicon)\n")

    predictions = []
    ground_truths = []

    for sample in tqdm(eval_samples, desc="Evaluating"):
        pred = run_inference(model, tokenizer, sample["input"])
        predictions.append(pred)
        ground_truths.append(json.loads(sample["output"]))

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    print_metrics(metrics)

    # Save results
    results_path = "./outputs/eval_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Detailed results saved to: {results_path}")

    print("\n" + "=" * 65)
    print("✅ Evaluation complete!")
    print("👉 When ready: cd ../02_qlora && read README.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
