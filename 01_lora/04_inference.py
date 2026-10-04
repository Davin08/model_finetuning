"""
04_inference.py — Phase 1: Run your trained LoRA model

Load the trained adapter and extract structured data from new payroll text.
Run: python 04_inference.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "./outputs/lora_phi2_payroll/final_adapter"

INSTRUCTION = (
    "You are a payroll document parser. Extract all relevant financial fields from "
    "the following payroll document text and return them as a valid JSON object. "
    "Include document_type, employee information, all monetary amounts, and dates. "
    "Return ONLY the JSON, no explanation."
)

# ── Test payroll documents ────────────────────────────────────────────────────
TEST_DOCUMENTS = [
    {
        "name": "Test 1: Standard Pay Stub",
        "text": """TechCorp Inc.
PAY STUB
Employee: Alex Rivera
Employee ID: EMP-55432
Department: Engineering
Pay Period: 10/01/2024 - 10/15/2024
Pay Frequency: Semi-Monthly

EARNINGS
  Regular Pay:                $4,583.33
  Total Gross Pay:            $4,583.33

DEDUCTIONS
  Federal Income Tax:        -$687.50
  State Income Tax:          -$229.17
  Social Security (6.2%):    -$284.17
  Medicare (1.45%):          -$66.46
  Health Insurance:          -$200.00
  401(k) Contribution:       -$229.17
  Total Deductions:          -$1,696.47

NET PAY: $2,886.86
""",
    },
    {
        "name": "Test 2: W-2 Form",
        "text": """W-2 Wage and Tax Statement 2023

Employer: Summit Financial Group
Employee Name: Jordan Kim
Employee SSN: XXX-XX-7821

Box 1  - Wages, tips, other compensation: $87,000.00
Box 2  - Federal income tax withheld: $15,660.00
Box 3  - Social security wages: $87,000.00
Box 4  - Social security tax withheld: $5,394.00
Box 5  - Medicare wages and tips: $87,000.00
Box 6  - Medicare tax withheld: $1,261.50
Box 16 - State wages: $87,000.00
Box 17 - State income tax: $4,785.00
Tax Year: 2023
""",
    },
]


def load_model():
    """Load base model + LoRA adapter."""
    import os
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter not found at {ADAPTER_PATH}")
        print("   Run training first: python 03_train_lora.py")
        return None, None

    print(f"⏳ Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print(f"⏳ Loading LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Model ready on {device}")
    return model, tokenizer


def extract_payroll(model, tokenizer, document_text: str, max_new_tokens: int = 512):
    """Run inference on a payroll document."""
    prompt = (
        f"Instruct: {INSTRUCTION}\n\n"
        f"Document:\n{document_text.strip()}\n\n"
        f"Output:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,       # Low temperature = more deterministic
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the input)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return output_text.strip()


def parse_json_output(output_text: str):
    """Try to extract valid JSON from model output."""
    # Find the first { and last }
    start = output_text.find("{")
    end = output_text.rfind("}") + 1

    if start == -1 or end == 0:
        return None, "No JSON braces found"

    json_str = output_text[start:end]
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, str(e)


def main():
    print("=" * 65)
    print("  Phase 1: LoRA Inference — Payroll Extraction")
    print("=" * 65)

    model, tokenizer = load_model()
    if model is None:
        return

    for i, test_doc in enumerate(TEST_DOCUMENTS, 1):
        print(f"\n{'─' * 65}")
        print(f"🧪 {test_doc['name']}")
        print("─" * 65)
        print("📄 Input document:")
        print(test_doc["text"][:300])
        if len(test_doc["text"]) > 300:
            print("  ...")

        print("\n🤖 Extracting with fine-tuned model...")
        raw_output = extract_payroll(model, tokenizer, test_doc["text"])

        print(f"\n📤 Raw model output:\n{raw_output[:600]}")

        # Try to parse as JSON
        parsed, error = parse_json_output(raw_output)
        if parsed:
            print(f"\n✅ Successfully parsed JSON!")
            print(f"   Fields extracted: {list(parsed.keys())}")
            print(f"\n   Key values:")
            for field in ["employee_name", "gross_pay", "net_pay", "pay_period_start", "document_type"]:
                if field in parsed:
                    print(f"   {field}: {parsed[field]}")
        else:
            print(f"\n⚠️  JSON parsing failed: {error}")
            print("   Model may need more training or prompt adjustment")

    print("\n" + "=" * 65)
    print("✅ Inference complete!")
    print("👉 Next: python 05_evaluate.py  — measure accuracy on eval set")
    print("=" * 65)


if __name__ == "__main__":
    main()
