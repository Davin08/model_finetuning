"""
01_tokenization_demo.py — Phase 0: Understanding Tokenization

This is the FIRST step in every LLM pipeline.
We'll use payroll-like text so it's relevant to our project.

Run: python 01_tokenization_demo.py
"""

from transformers import AutoTokenizer

# We'll use GPT-2's tokenizer for this demo (fast to download, no auth needed)
# Later phases will use Llama / Phi tokenizers
MODEL_NAME = "gpt2"

def demo_basic_tokenization(tokenizer):
    """Show how text becomes tokens (numbers)."""
    print("\n" + "─" * 60)
    print("📝 DEMO 1: Basic Tokenization")
    print("─" * 60)

    examples = [
        "Hello, world!",
        "Employee Name: John Smith",
        "Gross Pay: $4,200.00",
        "Federal Tax Withheld: $630.00",
        "Pay Period: 09/01/2024 - 09/15/2024",
    ]

    for text in examples:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        print(f"\nText:   {repr(text)}")
        print(f"Tokens: {tokens}")
        print(f"IDs:    {ids}")
        print(f"Count:  {len(tokens)} tokens")


def demo_special_tokens(tokenizer):
    """Show special tokens used in chat/instruction models."""
    print("\n" + "─" * 60)
    print("🔖 DEMO 2: Special Tokens")
    print("─" * 60)
    print("""
Special tokens are markers the model uses to understand structure:
  <|endoftext|>  — marks end of a document
  <s>, </s>      — start/end of sequence (used in Llama)
  [INST], [/INST] — instruction markers (used in Mistral)
  <|user|>       — marks user turn (used in Phi)
  <|assistant|>  — marks model turn (used in Phi)

These are critical in fine-tuning! The model learns to:
  1. Recognize [INST] → "I'm being given instructions"
  2. Generate after [/INST] → "Now I respond"
""")
    print(f"GPT-2's special tokens: {tokenizer.all_special_tokens}")
    print(f"EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")


def demo_encoding_decoding(tokenizer):
    """Show the full encode → decode roundtrip."""
    print("\n" + "─" * 60)
    print("🔄 DEMO 3: Encode → Decode Roundtrip")
    print("─" * 60)

    payroll_text = """
Pay Stub - Q3 2024
Employee: Sarah Johnson
Employee ID: EMP-10234
Pay Period: 09/01/2024 to 09/15/2024
Gross Pay: $5,416.67
Federal Tax: -$812.50
State Tax: -$216.67
Social Security: -$335.83
Medicare: -$78.54
Net Pay: $3,973.13
"""

    print("Original text:")
    print(payroll_text)

    # Encode to token IDs
    token_ids = tokenizer.encode(payroll_text)
    print(f"Token count: {len(token_ids)} tokens")
    print(f"First 20 token IDs: {token_ids[:20]}")

    # Decode back to text
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded back to text (identical): {decoded == payroll_text}")


def demo_vocabulary_size(tokenizer):
    """Show vocabulary size and what that means."""
    print("\n" + "─" * 60)
    print("📚 DEMO 4: Vocabulary Size")
    print("─" * 60)
    vocab_size = tokenizer.vocab_size
    print(f"""
Vocabulary size: {vocab_size:,} unique tokens

This means the model's final layer outputs {vocab_size:,} probabilities —
one for each possible next token.

Common vocab sizes:
  GPT-2:          50,257 tokens
  Llama-3:       128,256 tokens  (← we'll use this later)
  Phi-3:          32,064 tokens

Larger vocab = better handling of rare words, numbers, and symbols
— important for payroll data with lots of numbers and codes!
""")


def demo_numbers_and_currency(tokenizer):
    """
    Key insight: numbers are often split into many tokens.
    This matters for payroll extraction!
    """
    print("\n" + "─" * 60)
    print("💰 DEMO 5: How Numbers & Currency Are Tokenized")
    print("─" * 60)
    print("⚠️  This is IMPORTANT for payroll projects!\n")

    examples = [
        "$4,200.00",
        "$4200",
        "4200.00",
        "123456789",
        "09/15/2024",
        "SSN: 123-45-6789",
    ]

    for text in examples:
        tokens = tokenizer.tokenize(text)
        print(f"  {repr(text):25} → {tokens}  ({len(tokens)} tokens)")

    print("""
Key insight: Numbers are often split into many tokens!
"$4,200.00" → 5+ tokens

This means:
1. Your model needs context to understand these are ONE value
2. Fine-tuning on payroll data teaches it to group these correctly
3. Asking the model to output JSON with numbers works well because
   it learns the full number pattern, not just individual digits
""")


def main():
    print("=" * 60)
    print("  Phase 0: Tokenization Demo")
    print("  Powered by HuggingFace Transformers")
    print("=" * 60)

    print(f"\n⏳ Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Tokenizer loaded!")

    demo_basic_tokenization(tokenizer)
    demo_special_tokens(tokenizer)
    demo_encoding_decoding(tokenizer)
    demo_vocabulary_size(tokenizer)
    demo_numbers_and_currency(tokenizer)

    print("\n" + "=" * 60)
    print("🎉 Tokenization demo complete!")
    print("""
Key takeaways:
  1. Text → Tokens → IDs (numbers the model processes)
  2. Special tokens ([INST], <|user|>) structure conversations
  3. Numbers and currency split into multiple tokens
  4. Vocabulary size = output probabilities at each step

✅ You've completed Phase 0!
👉 Next: cd ../01_lora && read README.md
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
