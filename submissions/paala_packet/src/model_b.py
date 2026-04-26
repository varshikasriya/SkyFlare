# model_b.py — v2 (improved for sparse fire pixels)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from configs.model_config import CONFIG_B as CONFIG

print(f"Training on: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

INPUT_FEATURES = [
    'elevation', 'pdsi', 'NDVI', 'pr', 'sph',
    'th', 'tmmn', 'tmmx', 'vs', 'erc', 'population', 'PrevFireMask'
]
TARGET_FEATURE = 'FireMask'
NUM_CHANNELS   = len(INPUT_FEATURES)

FEATURE_SCHEMA = {
    f: tf.io.FixedLenFeature([64*64], tf.float32)
    for f in INPUT_FEATURES + [TARGET_FEATURE]
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_tfrecords(data_dir, split="train"):
    files = sorted(Path(data_dir).glob(f"*{split}*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No {split} .tfrecord files in {data_dir}")
    print(f"Loading {len(files)} {split} files...")

    all_inputs, all_targets = [], []
    ds = tf.data.TFRecordDataset([str(f) for f in files])

    for i, raw in enumerate(ds):
        sample = tf.io.parse_single_example(raw, FEATURE_SCHEMA)
        inp = np.stack(
            [sample[f].numpy().reshape(64, 64) for f in INPUT_FEATURES], axis=0
        ).astype(np.float32)
        tgt = sample[TARGET_FEATURE].numpy().reshape(64, 64)
        tgt = np.where(tgt >= 1, 1.0, 0.0).astype(np.float32)

        # Keep ALL samples that have prev fire (don't skip no-fire ones —
        # the model needs to learn "prev fire but no spread" too)
        if inp[11].max() == 0 and tgt.max() == 0:
            continue

        all_inputs.append(inp)
        all_targets.append(tgt)
        if (i + 1) % 3000 == 0:
            print(f"  {i+1} samples loaded...")

    print(f"  → {len(all_inputs)} usable {split} samples")
    return np.array(all_inputs, dtype=np.float32), np.array(all_targets, dtype=np.float32)

def compute_stats(inputs):
    means = inputs.mean(axis=(0, 2, 3))
    stds  = inputs.std(axis=(0, 2, 3))
    stds  = np.where(stds < 1e-6, 1.0, stds)
    return means, stds

# ── DATASET WITH OVERSAMPLING WEIGHTS ─────────────────────────────────────────
class WildfireDataset(Dataset):
    def __init__(self, inputs, targets, means, stds, augment=False):
        self.inputs  = torch.tensor(inputs,  dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        self.means   = torch.tensor(means,   dtype=torch.float32).view(-1, 1, 1)
        self.stds    = torch.tensor(stds,    dtype=torch.float32).view(-1, 1, 1)
        self.augment = augment

        # Compute per-sample weight: samples with more fire pixels get higher weight
        fire_counts = targets.sum(axis=(1, 2))  # (N,)
        # Weight = 1 for no-fire, up to 10x for heavy-fire samples
        self.sample_weights = 1.0 + 9.0 * (fire_counts / (fire_counts.max() + 1e-6))
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        x = (x - self.means) / self.stds

        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            # Random vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            # Random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])

        return x, y

# ── MODEL: U-Net ──────────────────────────────────────────────────────────────
class WildfireCNN(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch,  out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out     = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.out(d1))

# ── LOSS: Stronger Focal + Dice ───────────────────────────────────────────────
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt  = torch.where(target == 1, pred, 1 - pred)
        w   = self.alpha * (1 - pt) ** self.gamma
        return (w * bce).mean()

    def dice_loss(self, pred, target):
        smooth = 1e-6
        p = pred.view(-1); t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth)

    def forward(self, pred, target):
        return self.focal_loss(pred, target) + self.dice_loss(pred, target)

# ── METRICS (use lower threshold) ─────────────────────────────────────────────
def calculate_iou(pred, target, threshold=0.3):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    if union == 0:
        return 1.0
    return (inter / union).item()

def calculate_f1(pred, target, threshold=0.3):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).item()

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train():
    Path(CONFIG['model_save_dir']).mkdir(exist_ok=True)
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)

    print("Loading training data...")
    train_inputs, train_targets = load_tfrecords(CONFIG['data_dir'], split="train")
    print("Loading eval data...")
    val_inputs, val_targets = load_tfrecords(CONFIG['data_dir'], split="eval")

    means, stds = compute_stats(train_inputs)
    np.save("models/wildfire_means.npy", means)
    np.save("models/wildfire_stds.npy", stds)

    print(f"\nFire pixel fraction: {train_targets.mean()*100:.2f}%")
    print(f"Train: {len(train_inputs)} | Val: {len(val_inputs)}")

    train_ds = WildfireDataset(train_inputs, train_targets, means, stds, augment=True)
    val_ds   = WildfireDataset(val_inputs,   val_targets,   means, stds, augment=False)

    # Weighted sampler: oversample fire-heavy patches
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=True)

    model = WildfireCNN(in_channels=NUM_CHANNELS).to(CONFIG['device'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    # ReduceLROnPlateau: halve LR if val IoU stops improving for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    criterion = FocalDiceLoss(alpha=0.85, gamma=3.0)

    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}
    best_iou = 0.0
    no_improve = 0

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print(f"Using IoU threshold: {CONFIG['iou_threshold']}")
    print("=" * 70)

    for epoch in range(CONFIG['num_epochs']):

        # Train
        model.train()
        train_losses = []
        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{CONFIG['num_epochs']} [Train]"):
            imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_ious, val_f1s = [], [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Ep {epoch+1:02d}/{CONFIG['num_epochs']} [Val]  "):
                imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
                preds = model(imgs)
                val_losses.append(criterion(preds, masks).item())
                val_ious.append(calculate_iou(preds, masks, threshold=CONFIG['iou_threshold']))
                val_f1s.append(calculate_f1(preds, masks,   threshold=CONFIG['iou_threshold']))

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        avg_iou   = np.mean(val_ious)
        avg_f1    = np.mean(val_f1s)

        scheduler.step(avg_iou)  # plateau scheduler watches IoU

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_iou'].append(avg_iou)
        history['val_f1'].append(avg_f1)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:02d} | Loss {avg_train:.4f}→{avg_val:.4f} | IoU: {avg_iou:.4f} | F1: {avg_f1:.4f} | LR: {lr_now:.2e}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            no_improve = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'best_iou':         best_iou,
                'means':            means,
                'stds':             stds,
                'iou_threshold':    CONFIG['iou_threshold'],
                'config':           CONFIG,
            }, "models/best_wildfire_model.pth")
            print(f"  ✓ New best! IoU: {best_iou:.4f}")
        else:
            no_improve += 1

        # Early stop if no improvement for 15 epochs
        if no_improve >= 15:
            print(f"\nEarly stopping at epoch {epoch+1} — no improvement for 15 epochs.")
            break

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['val_iou'], color='green')
    axes[1].axhline(best_iou, color='red', linestyle='--', label=f'Best: {best_iou:.4f}')
    axes[1].set_title(f'Val IoU (threshold={CONFIG["iou_threshold"]})'); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history['val_f1'], color='blue')
    axes[2].set_title('Val F1'); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(Path(CONFIG['output_dir']) / "wildfire_training_curves.png", dpi=150)
    print(f"\nDone! Best IoU: {best_iou:.4f}")
    print("Model saved to models/best_wildfire_model.pth")


if __name__ == "__main__":
    import traceback, sys
    try:
        train()
    except BaseException as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)