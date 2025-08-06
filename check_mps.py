#!/usr/bin/env python3
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("Testing MPS device...")
    try:
        device = torch.device("mps")
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.mm(test_tensor, test_tensor.T)
        print("✅ MPS test successful!")
        print(f"Test tensor shape: {result.shape}")
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
else:
    print("❌ MPS not available on this system")

# Also check CUDA for completeness
print(f"CUDA available: {torch.cuda.is_available()}")