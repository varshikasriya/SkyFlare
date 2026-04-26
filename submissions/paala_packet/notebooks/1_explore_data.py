# 2_explore_data.py
# Explore the HLS Burn Scars dataset before training

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent  # notebooks/ → wildfire-spread/
DATA_DIR  = ROOT / "data" / "hls_burn_scars"
TRAIN_DIR = DATA_DIR / "training"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── CHECK DATA EXISTS ───────────────────────────────────────────────────────
if not DATA_DIR.exists():
    print("ERROR: Data not found at data/hls_burn_scars/")
    print("Download it first using the download script.")
    exit()

# ── COUNT FILES ─────────────────────────────────────────────────────────────
# 1. Get all .tif files
all_tifs = list(TRAIN_DIR.glob("*.tif"))

# 2. Separate them based on the filename content
mask_files = sorted([f for f in all_tifs if ".mask" in f.name])
img_files = sorted([f for f in all_tifs if ".merged" in f.name])

# If .merged doesn't work, let's try a fallback
if len(img_files) == 0:
    img_files = sorted([f for f in all_tifs if ".mask" not in f.name])

print(f"Total .tif files found: {len(all_tifs)}")
print(f"Training images found: {len(img_files)}")
print(f"Training masks found:  {len(mask_files)}")

if len(img_files) == 0:
    print(f"\nDEBUG: First 3 files found in folder:")
    for f in all_tifs[:3]:
        print(f" - {f.name}")
    exit()

# ── LOAD ONE SAMPLE ─────────────────────────────────────────────────────────
sample_img_path = img_files[0]
sample_mask_path = mask_files[0]

print(f"\nLoading sample: {sample_img_path.name}")

img = rxr.open_rasterio(sample_img_path)
mask = rxr.open_rasterio(sample_mask_path)

print(f"Image shape: {img.shape}")   # should be (6, 512, 512)
print(f"Mask shape:  {mask.shape}")  # should be (1, 512, 512)
print(f"Image dtype: {img.dtype}")
print(f"Mask unique values: {np.unique(mask.values)}")  # should be -1, 0, 1

# Band info
band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
img_np = img.values  # shape: (6, 512, 512)
for i, name in enumerate(band_names):
    band = img_np[i]
    print(f"  Band {i+1} ({name}): min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}")

# Mask distribution
mask_np = mask.values[0]
print(f"\nMask distribution:")
print(f"  Burned (1):      {(mask_np == 1).sum()} pixels ({(mask_np == 1).mean()*100:.1f}%)")
print(f"  Not burned (0):  {(mask_np == 0).sum()} pixels ({(mask_np == 0).mean()*100:.1f}%)")
print(f"  No data (-1):    {(mask_np == -1).sum()} pixels ({(mask_np == -1).mean()*100:.1f}%)")

# ── VISUALIZE ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(f'HLS Burn Scars Sample: {sample_img_path.name}', fontsize=12)

# RGB composite (bands 2, 1, 0 = Red, Green, Blue)
rgb = img_np[[2, 1, 0]].transpose(1, 2, 0)
rgb = np.clip(rgb / rgb.max(), 0, 1)  # normalize for display
axes[0].imshow(rgb)
axes[0].set_title('RGB (True Color)')
axes[0].axis('off')

# NIR band
axes[1].imshow(img_np[3], cmap='YlGn')
axes[1].set_title('NIR Band (Vegetation)')
axes[1].axis('off')

# SWIR band (fire sensitive)
axes[2].imshow(img_np[4], cmap='hot')
axes[2].set_title('SWIR1 (Fire Sensitive)')
axes[2].axis('off')

# Burn mask
mask_display = np.where(mask_np == -1, np.nan, mask_np)
axes[3].imshow(mask_display, cmap='RdYlGn_r', vmin=0, vmax=1)
axes[3].set_title('Burn Mask\n(Red=Burned, Green=Safe)')
axes[3].axis('off')

plt.tight_layout()
save_path = OUTPUT_DIR / "sample_visualization.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {save_path}")
plt.show()

print("\nData exploration complete. Ready for training.")