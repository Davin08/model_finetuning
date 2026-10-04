"""
setup_check.py — Phase 0: Environment Setup for LLM Fine-Tuning on Apple Silicon

Run this first: python setup_check.py
"""

import subprocess
import sys
import importlib

REQUIRED_PACKAGES = [
    ("torch", "torch>=2.1.0"),
    ("transformers", "transformers>=4.40.0"),
    ("datasets", "datasets>=2.18.0"),
    ("peft", "peft>=0.10.0"),          # LoRA / QLoRA
    ("accelerate", "accelerate>=0.28.0"),
    ("bitsandbytes", "bitsandbytes>=0.43.0"),  # 4-bit quantization (QLoRA)
    ("trl", "trl>=0.8.0"),             # Training utilities (SFTTrainer, DPO)
    ("pypdf", "pypdf>=4.0.0"),         # PDF parsing for our payroll project
    ("sentencepiece", "sentencepiece>=0.1.99"),
    ("evaluate", "evaluate>=0.4.0"),
    ("rouge_score", "rouge-score>=0.1.2"),
]

OPTIONAL_PACKAGES = [
    ("wandb", "wandb>=0.16.0"),        # Experiment tracking (optional)
]

def check_python():
    version = sys.version_info
    major, minor = version.major, version.minor
    status = "✅" if (major == 3 and minor >= 10) else "❌"
    print(f"{status} Python {major}.{minor} (need 3.10+)")
    return major == 3 and minor >= 10

def check_package(module_name, install_name):
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {module_name} ({version})")
        return True
    except ImportError:
        print(f"  ❌ {module_name} — not installed")
        return False

def check_mps():
    try:
        import torch
        available = torch.backends.mps.is_available()
        built = torch.backends.mps.is_built()
        if available and built:
            print("  ✅ MPS (Apple GPU) — available and built")
            return True
        elif built and not available:
            print("  ⚠️  MPS built but not available (running on Intel Mac or VM?)")
            return False
        else:
            print("  ❌ MPS not available — will fall back to CPU (slower)")
            return False
    except ImportError:
        print("  ❌ torch not installed — can't check MPS")
        return False

def install_missing(missing):
    if not missing:
        return
    print("\n📦 Installing missing packages...")
    for pkg in missing:
        print(f"  Installing {pkg}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Installed {pkg}")
        else:
            print(f"  ❌ Failed to install {pkg}")
            print(f"     Error: {result.stderr[:200]}")

def main():
    print("=" * 60)
    print("  LLM Fine-Tuning — Environment Setup Check")
    print("  Phase 0: Foundations")
    print("=" * 60)

    # Python version
    print("\n🐍 Python Version:")
    py_ok = check_python()

    # Required packages
    print("\n📦 Required Packages:")
    missing = []
    for module_name, install_name in REQUIRED_PACKAGES:
        ok = check_package(module_name, install_name)
        if not ok:
            missing.append(install_name)

    # Optional packages
    print("\n📦 Optional Packages (recommended):")
    for module_name, install_name in OPTIONAL_PACKAGES:
        check_package(module_name, install_name)

    # MPS check
    print("\n🖥️  Hardware Acceleration:")
    mps_ok = check_mps()

    # Summary
    print("\n" + "=" * 60)
    if missing:
        print(f"⚠️  Missing {len(missing)} required package(s).")
        print("   Run the following to install them:")
        print(f"\n   pip install {' '.join(missing)}\n")
        print("   Or let this script install them for you? (y/n): ", end="")
        response = input().strip().lower()
        if response == "y":
            install_missing(missing)
            print("\n✅ Done! Re-run this script to verify.")
    else:
        print("✅ All required packages are installed!")
        if mps_ok:
            print("✅ MPS GPU acceleration is available.")
            print("\n🎉 Your environment is ready for LLM fine-tuning!")
        else:
            print("⚠️  No GPU acceleration — training will be CPU-only (very slow).")
            print("   Consider using Google Colab for the heavier training exercises.")
        print("\n👉 Next: python verify_mps.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
