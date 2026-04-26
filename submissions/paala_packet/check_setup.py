# check_setup.py
# Run this first to confirm your environment is ready

import sys

print(f"Python version: {sys.version}") # Should be 3.11 or 3.12

# Check PyTorch
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU found. Training will be very slow.")
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")

# Check TerraTorch
try:
    import terratorch

    print(f"TerraTorch: OK")
except ImportError:
    print("ERROR: TerraTorch not installed. Run: pip install terratorch>=1.2.4")

# Check rioxarray
try:
    import rioxarray

    print(f"rioxarray: OK")
except ImportError:
    print("ERROR: rioxarray not installed.")

# Check diffusers
try:
    import diffusers

    print(f"diffusers version: {diffusers.__version__}")
except ImportError:
    print("ERROR: diffusers not installed. Run: pip install diffusers==0.30.0")

# Check data folders exist
import os

folders = ['data', 'models', 'outputs']
for folder in folders:
    if os.path.exists(folder):
        print(f"Folder '{folder}': OK")
    else:
        os.makedirs(folder)
        print(f"Folder '{folder}': Created")

# Try loading TerraMind (this will download weights ~300MB on first run)
print("\nAttempting to load TerraMind-small...")
print("(This downloads ~300MB on first run - wait for it)")
try:
    from terratorch import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        'terramind_v1_small',
        pretrained=True,
        modalities=['S2L2A']
    )
    print("TerraMind-small loaded successfully!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

except Exception as e:
    print(f"ERROR loading TerraMind: {e}")

print("\nSetup check complete.")