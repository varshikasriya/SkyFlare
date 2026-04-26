# 2_explore_tfrecord.py
# Explore the Next Day Wildfire Spread dataset (.tfrecord format)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG — UPDATE THIS PATH ──────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
TFRECORD_DIR = ROOT / "data" / "ndws"
OUTPUT_DIR   = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── FEATURE SCHEMA ──────────────────────────────────────────────────────────
# The Next Day Wildfire Spread dataset has these known features:
FEATURES = {
    # Input bands (each is a 64x64 float32 patch)
    'elevation':        tf.io.FixedLenFeature([64*64], tf.float32),
    'pdsi':             tf.io.FixedLenFeature([64*64], tf.float32),  # drought index
    'NDVI':             tf.io.FixedLenFeature([64*64], tf.float32),  # vegetation
    'pr':               tf.io.FixedLenFeature([64*64], tf.float32),  # precipitation
    'sph':              tf.io.FixedLenFeature([64*64], tf.float32),  # humidity
    'th':               tf.io.FixedLenFeature([64*64], tf.float32),  # wind direction
    'tmmn':             tf.io.FixedLenFeature([64*64], tf.float32),  # min temp
    'tmmx':             tf.io.FixedLenFeature([64*64], tf.float32),  # max temp
    'vs':               tf.io.FixedLenFeature([64*64], tf.float32),  # wind speed
    'erc':              tf.io.FixedLenFeature([64*64], tf.float32),  # energy release component
    'population':       tf.io.FixedLenFeature([64*64], tf.float32),
    'PrevFireMask':     tf.io.FixedLenFeature([64*64], tf.float32),  # fire at T
    # Target
    'FireMask':         tf.io.FixedLenFeature([64*64], tf.float32),  # fire at T+1
}

INPUT_FEATURES = ['elevation', 'pdsi', 'NDVI', 'pr', 'sph', 'th', 'tmmn', 'tmmx', 'vs', 'erc', 'population', 'PrevFireMask']
TARGET_FEATURE = 'FireMask'

def parse_record(example_proto):
    return tf.io.parse_single_example(example_proto, FEATURES)

# ── FIND FILES ──────────────────────────────────────────────────────────────
tfrecord_files = list(TFRECORD_DIR.glob("*.tfrecord"))
print(f"Found {len(tfrecord_files)} .tfrecord files:")
for f in tfrecord_files[:5]:
    print(f"  {f.name}")

if not tfrecord_files:
    print("\nERROR: No .tfrecord files found!")
    print(f"Make sure your files are in: {TFRECORD_DIR.absolute()}")
    exit()

# ── LOAD AND INSPECT ────────────────────────────────────────────────────────
dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files[:1]])
dataset = dataset.map(parse_record)

print("\nLoading first sample...")
for sample in dataset.take(1):
    print("\nFeatures found:")
    for key in INPUT_FEATURES:
        vals = sample[key].numpy()
        print(f"  {key:20s}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

    fire_mask = sample[TARGET_FEATURE].numpy().reshape(64, 64)
    print(f"\nFireMask (target):")
    print(f"  Shape after reshape: {fire_mask.shape}")
    print(f"  Unique values: {np.unique(fire_mask)}")
    print(f"  Fire pixels (1): {(fire_mask == 1).sum()} / {fire_mask.size} ({(fire_mask==1).mean()*100:.1f}%)")

    # ── VISUALIZE ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Next Day Wildfire Dataset - Sample', fontsize=14)

    show_features = ['elevation', 'NDVI', 'vs', 'erc', 'tmmx', 'sph', 'PrevFireMask', 'FireMask']
    cmaps = ['terrain', 'YlGn', 'Blues', 'hot', 'RdYlBu_r', 'Blues', 'Reds', 'Reds']

    for ax, feat, cmap in zip(axes.flat, show_features, cmaps):
        data = sample[feat].numpy().reshape(64, 64)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(feat)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tfrecord_sample.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to outputs/tfrecord_sample.png")

# Count total samples
count = 0
full_dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
for _ in full_dataset:
    count += 1
print(f"\nTotal samples across all tfrecord files: {count}")
print("\nExploration complete!")