from __future__ import annotations
import platform
import sys
import torch

print("Python:", sys.version.replace("\n", " "))
print("OS:", platform.platform())
print("PyTorch:", torch.__version__)
print("Torchvision:", __import__("torchvision").__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"VRAM: {total:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda")
    print("CUDA test:", float((x @ x).mean()))
else:
    raise SystemExit("ОШИБКА: PyTorch не видит NVIDIA GPU. Обнови драйвер NVIDIA и повтори setup.")
print("\nВСЕ ГОТОВО: CUDA работает.")
