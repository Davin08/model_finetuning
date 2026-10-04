"""
02_generate_synthetic_data.py — Phase 1: Create Payroll Training Data

We generate synthetic (fake but realistic) payroll data so we can
start training without needing real PDFs. In Phase 5 (Capstone), we'll
replace this with your actual payroll PDFs.

Run: python 02_generate_synthetic_data.py
"""

import json
import random
import os
from datetime import date, timedelta

# ── Data pools ────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "James", "Maria", "John", "Sarah", "Michael", "Jennifer", "Robert",
    "Linda", "David", "Patricia", "Emma", "William", "Olivia", "Richard",
    "Sofia", "Carlos", "Priya", "Kenji", "Amara", "Hassan",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Martinez",
    "Davis", "Rodriguez", "Lopez", "Lee", "Harris", "Walker", "Patel",
    "Kim", "Chen", "Nguyen", "Thompson", "White", "Jackson",
]
COMPANIES = [
    "Acme Corporation", "Pinnacle Staffing", "BlueSky Solutions",
    "Vertex Industries", "Summit Financial Group", "Pacific Coast Logistics",
    "Metro Health Systems", "Lakeside Technologies", "Granite Construction Co.",
    "Maple Leaf Consulting",
]
DEPARTMENTS = [
    "Engineering", "Sales", "Human Resources", "Finance", "Operations",
    "Marketing", "Customer Success", "Legal", "IT", "Supply Chain",
]
PAY_TYPES = ["Biweekly", "Weekly", "Semi-Monthly", "Monthly"]
DOC_TYPES = [
    "pay_stub",      # Regular paycheck stub
    "w2",            # Annual W-2 form
    "1099_nec",      # Contractor 1099
    "direct_deposit_notice",  # Direct deposit confirmation
]

# ── Helper functions ──────────────────────────────────────────────────────────

def random_date_range(pay_type: str):
    """Generate a realistic pay period based on pay frequency."""
    start = date(2024, random.randint(1, 12), random.choice([1, 15]))
    if pay_type == "Weekly":
        end = start + timedelta(days=6)
    elif pay_type == "Biweekly":
        end = start + timedelta(days=13)
    elif pay_type == "Semi-Monthly":
        end = start + timedelta(days=14)
    else:  # Monthly
        end = start + timedelta(days=29)
    return start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y")


def calculate_taxes(gross: float):
    """Simplified tax calculation (not financial advice!)."""
    federal = round(gross * 0.15, 2)   # ~15% effective rate
    state   = round(gross * 0.05, 2)   # ~5%
    ss      = round(gross * 0.062, 2)  # Social Security 6.2%
    medicare = round(gross * 0.0145, 2) # Medicare 1.45%
    return federal, state, ss, medicare


def generate_pay_stub(doc_id: int):
    """Generate one complete pay stub as text + structured JSON labels."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    emp_id = f"EMP-{random.randint(10000, 99999)}"
    company = random.choice(COMPANIES)
    department = random.choice(DEPARTMENTS)
    pay_type = random.choice(PAY_TYPES)
    period_start, period_end = random_date_range(pay_type)

    # Gross pay (annual salary / pay periods)
    annual_salary = random.choice(range(35000, 150001, 5000))
    periods_per_year = {"Weekly": 52, "Biweekly": 26, "Semi-Monthly": 24, "Monthly": 12}[pay_type]
    gross = round(annual_salary / periods_per_year, 2)

    # Deductions
    federal, state, ss, medicare = calculate_taxes(gross)
    health = random.choice([0, 125.00, 200.00, 350.00])
    retirement_401k = round(gross * random.choice([0, 0.03, 0.05, 0.06]), 2)

    total_deductions = round(federal + state + ss + medicare + health + retirement_401k, 2)
    net_pay = round(gross - total_deductions, 2)

    # Generate the raw text (what you'd extract from a PDF)
    raw_text = f"""{company}
PAY STUB
Employee: {name}
Employee ID: {emp_id}
Department: {department}
Pay Period: {period_start} - {period_end}
Pay Frequency: {pay_type}

EARNINGS
  Regular Pay:                ${gross:,.2f}
  Total Gross Pay:            ${gross:,.2f}

DEDUCTIONS
  Federal Income Tax:        -${federal:,.2f}
  State Income Tax:          -${state:,.2f}
  Social Security (6.2%):    -${ss:,.2f}
  Medicare (1.45%):          -${medicare:,.2f}
  Health Insurance:          -${health:,.2f}
  401(k) Contribution:       -${retirement_401k:,.2f}
  Total Deductions:          -${total_deductions:,.2f}

NET PAY: ${net_pay:,.2f}

YTD (Year-to-Date) Gross: ${gross * random.randint(1, periods_per_year):,.2f}
"""

    # The structured output we want the model to produce
    structured_output = {
        "document_type": "pay_stub",
        "employee_name": name,
        "employee_id": emp_id,
        "company": company,
        "department": department,
        "pay_period_start": period_start,
        "pay_period_end": period_end,
        "pay_frequency": pay_type,
        "gross_pay": gross,
        "federal_tax": federal,
        "state_tax": state,
        "social_security": ss,
        "medicare": medicare,
        "health_insurance": health,
        "retirement_401k": retirement_401k,
        "total_deductions": total_deductions,
        "net_pay": net_pay,
    }

    return raw_text, structured_output


def generate_w2(doc_id: int):
    """Generate a W-2 form sample."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    ssn_last4 = random.randint(1000, 9999)
    company = random.choice(COMPANIES)
    year = 2023

    wages = random.choice(range(30000, 200001, 1000))
    federal_withheld = round(wages * 0.18, 2)
    state_withheld = round(wages * 0.055, 2)
    ss_wages = min(wages, 160200)  # 2023 SS wage base
    ss_withheld = round(ss_wages * 0.062, 2)
    medicare_withheld = round(wages * 0.0145, 2)

    raw_text = f"""W-2 Wage and Tax Statement {year}

Employer: {company}
Employee Name: {name}
Employee SSN: XXX-XX-{ssn_last4}

Box 1  - Wages, tips, other compensation: ${wages:,.2f}
Box 2  - Federal income tax withheld: ${federal_withheld:,.2f}
Box 3  - Social security wages: ${ss_wages:,.2f}
Box 4  - Social security tax withheld: ${ss_withheld:,.2f}
Box 5  - Medicare wages and tips: ${wages:,.2f}
Box 6  - Medicare tax withheld: ${medicare_withheld:,.2f}
Box 16 - State wages: ${wages:,.2f}
Box 17 - State income tax: ${state_withheld:,.2f}
Tax Year: {year}
"""

    structured_output = {
        "document_type": "w2",
        "tax_year": year,
        "employee_name": name,
        "employee_ssn_last4": str(ssn_last4),
        "employer": company,
        "wages": wages,
        "federal_tax_withheld": federal_withheld,
        "social_security_wages": ss_wages,
        "social_security_withheld": ss_withheld,
        "medicare_wages": wages,
        "medicare_withheld": medicare_withheld,
        "state_wages": wages,
        "state_tax_withheld": state_withheld,
    }

    return raw_text, structured_output


def format_as_training_sample(raw_text: str, structured_output: dict, doc_type: str):
    """Format as an instruction-following training sample."""
    instruction = (
        "You are a payroll document parser. Extract all relevant financial fields from "
        "the following payroll document text and return them as a valid JSON object. "
        "Include document_type, employee information, all monetary amounts, and dates. "
        "Return ONLY the JSON, no explanation."
    )

    return {
        "instruction": instruction,
        "input": raw_text.strip(),
        "output": json.dumps(structured_output, indent=2),
        "doc_type": doc_type,
    }


def generate_dataset(n_samples: int = 300):
    """Generate a full training + evaluation dataset."""
    samples = []

    # 70% pay stubs, 30% W-2s (mix document types)
    n_stubs = int(n_samples * 0.7)
    n_w2s = n_samples - n_stubs

    print(f"  Generating {n_stubs} pay stubs...")
    for i in range(n_stubs):
        raw_text, labels = generate_pay_stub(i)
        sample = format_as_training_sample(raw_text, labels, "pay_stub")
        samples.append(sample)

    print(f"  Generating {n_w2s} W-2 forms...")
    for i in range(n_w2s):
        raw_text, labels = generate_w2(i)
        sample = format_as_training_sample(raw_text, labels, "w2")
        samples.append(sample)

    # Shuffle
    random.shuffle(samples)

    # Split 85% train, 15% eval
    split = int(len(samples) * 0.85)
    train = samples[:split]
    eval_ = samples[split:]

    return train, eval_


def main():
    print("=" * 60)
    print("  Phase 1: Generating Synthetic Payroll Dataset")
    print("=" * 60)

    random.seed(42)
    os.makedirs("data", exist_ok=True)

    print("\n📋 Generating 300 samples...")
    train_data, eval_data = generate_dataset(n_samples=300)

    # Save as JSONL (one JSON object per line — standard ML format)
    train_path = "data/train.jsonl"
    eval_path = "data/eval.jsonl"

    with open(train_path, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")

    with open(eval_path, "w") as f:
        for sample in eval_data:
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ Saved {len(train_data)} training samples → {train_path}")
    print(f"✅ Saved {len(eval_data)} eval samples → {eval_path}")

    # Show a sample
    print("\n" + "─" * 60)
    print("📄 Sample training record:")
    print("─" * 60)
    sample = train_data[0]
    print(f"Instruction: {sample['instruction'][:80]}...")
    print(f"\nInput (first 200 chars):\n{sample['input'][:200]}...")
    print(f"\nOutput (JSON):\n{sample['output'][:300]}...")

    # Dataset stats
    print("\n" + "─" * 60)
    print("📊 Dataset Statistics:")
    doc_types = [s["doc_type"] for s in train_data + eval_data]
    from collections import Counter
    counts = Counter(doc_types)
    for doc_type, count in counts.items():
        print(f"  {doc_type}: {count} samples")

    avg_input_len = sum(len(s["input"].split()) for s in train_data) / len(train_data)
    avg_output_len = sum(len(s["output"].split()) for s in train_data) / len(train_data)
    print(f"\n  Avg input length:  {avg_input_len:.0f} words")
    print(f"  Avg output length: {avg_output_len:.0f} words")

    print("\n" + "=" * 60)
    print("🎉 Dataset generated!")
    print("\n👉 Next: python 03_train_lora.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
