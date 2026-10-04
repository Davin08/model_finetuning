"""
verify_mps.py — Verify Apple Silicon MPS GPU acceleration

Run: python verify_mps.py
"""

import sys
import time

def verify_mps():
    print("=" * 55)
    print("  MPS (Metal Performance Shaders) Verification")
    print("=" * 55)

    # 1. Import torch
    try:
        import torch
        print(f"\n✅ PyTorch {torch.__version__} loaded")
    except ImportError:
        print("\n❌ PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # 2. Check MPS availability
    print("\n🔍 Checking MPS availability...")
    if not torch.backends.mps.is_built():
        print("❌ PyTorch was not built with MPS support")
        print("   Install Apple Silicon PyTorch: pip install torch")
        sys.exit(1)

    if not torch.backends.mps.is_available():
        print("❌ MPS is built but not available on this system")
        print("   Are you on an Intel Mac or VM? MPS requires Apple Silicon.")
        sys.exit(1)

    device = torch.device("mps")
    print("✅ MPS device is available!")

    # 3. Basic tensor operation on MPS
    print("\n🧪 Running basic tensor test on MPS...")
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y  # Matrix multiplication on GPU
        print(f"✅ Matrix multiply (1000x1000) on MPS — shape: {z.shape}")
    except Exception as e:
        print(f"❌ Tensor test failed: {e}")
        sys.exit(1)

    # 4. Speed benchmark — CPU vs MPS
    print("\n⚡ Speed benchmark: CPU vs MPS")
    size = 2000

    # CPU benchmark
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    start = time.time()
    for _ in range(5):
        _ = x_cpu @ y_cpu
    cpu_time = (time.time() - start) / 5
    print(f"   CPU:  {cpu_time * 1000:.1f} ms per matmul ({size}x{size})")

    # MPS benchmark
    x_mps = torch.randn(size, size, device=device)
    y_mps = torch.randn(size, size, device=device)
    torch.mps.synchronize()
    start = time.time()
    for _ in range(5):
        _ = x_mps @ y_mps
    torch.mps.synchronize()
    mps_time = (time.time() - start) / 5
    print(f"   MPS:  {mps_time * 1000:.1f} ms per matmul ({size}x{size})")

    speedup = cpu_time / mps_time
    print(f"   🚀 MPS speedup: {speedup:.1f}x faster than CPU")

    # 5. Memory info
    print(f"\n💾 MPS Memory:")
    print(f"   Allocated: {torch.mps.current_allocated_memory() / 1e6:.1f} MB")
    print(f"   Driver allocated: {torch.mps.driver_allocated_memory() / 1e6:.1f} MB")

    print("\n" + "=" * 55)
    print("✅ MPS is available! Training will use your GPU cores.")
    print("   Your Apple Silicon Mac is ready for LLM fine-tuning.")
    print("\n👉 Next: python 01_tokenization_demo.py")
    print("=" * 55)

if __name__ == "__main__":
    verify_mps()
