# model_a.py
# Fine-tune TerraMind-small on HLS Burn Scars for wildfire spread prediction

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import terratorch FIRST before any GDAL/rasterio DLLs load
from terratorch import BACKBONE_REGISTRY  # ← add this line here

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rioxarray as rxr                   # ← rioxarray loads AFTER
from tqdm import tqdm
import matplotlib.pyplot as plt
from configs.model_config import CONFIG_A as CONFIG

print(f"Training on: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── DATASET ─────────────────────────────────────────────────────────────────
class BurnScarsDataset(Dataset):
    """
        Loads HLS Burn Scars tiles and masks.
        Each sample = (image tensor, mask tensor)
        Image: 6 bands, normalized
        Mask:  binary (1=burned, 0=not burned), -1 values replaced with 0
    """
    def __init__(self, data_dir, split="training", image_size=224):
        self.image_size = image_size
        self.base_dir = Path(data_dir) / split

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.base_dir}")

        # Use the logic that worked in script #2
        all_tifs = list(self.base_dir.glob("*.tif"))
        self.img_files = sorted([f for f in all_tifs if "_merged" in f.name])
        self.mask_files = sorted([f for f in all_tifs if ".mask" in f.name])

        # Verification
        if len(self.img_files) != len(self.mask_files):
            print(f"Warning: Mismatch in {split}! Images: {len(self.img_files)}, Masks: {len(self.mask_files)}")

        print(f"  {split}: {len(self.img_files)} samples found")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        # Load image (6 bands)
        img = rxr.open_rasterio(img_path).values.astype(np.float32)

        # Load mask
        mask = rxr.open_rasterio(mask_path).values[0].astype(np.float32)

        # Handle no-data pixels (-1 → 0)
        mask = np.where(mask == -1, 0, mask)
        mask = np.clip(mask, 0, 1)

        # Resize to TerraMind's native 224x224
        img = self._resize(img, self.image_size)
        mask = self._resize_mask(mask, self.image_size)

        # Normalize
        img = self._normalize(img)

        # ── PAD 6 HLS BANDS → 12 S2L2A BANDS ──
        img_12 = np.zeros((12, self.image_size, self.image_size), dtype=np.float32)
        img_12[1] = img[0]  # B02 Blue
        img_12[2] = img[1]  # B03 Green
        img_12[3] = img[2]  # B04 Red
        img_12[8] = img[3]  # B8A NIR
        img_12[10] = img[4]  # B11 SWIR1
        img_12[11] = img[5]  # B12 SWIR2

        return torch.tensor(img_12, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def _resize(self, img, size):
        """Simple center crop or pad to target size"""
        _, h, w = img.shape
        # If image is larger, center crop
        if h >= size and w >= size:
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            return img[:, start_h:start_h+size, start_w:start_w+size]
        # If image is smaller, pad with zeros
        pad_img = np.zeros((img.shape[0], size, size), dtype=np.float32)
        pad_img[:, :h, :w] = img
        return pad_img

    def _resize_mask(self, mask, size):
        h, w = mask.shape
        if h >= size and w >= size:
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            return mask[start_h:start_h+size, start_w:start_w+size]
        pad_mask = np.zeros((size, size), dtype=np.float32)
        pad_mask[:h, :w] = mask
        return pad_mask

    def _normalize(self, img):
        """Normalize each band independently"""
        normalized = np.zeros_like(img)
        for i in range(img.shape[0]):
            band = img[i]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                normalized[i] = (band - band_min) / (band_max - band_min)
            else:
                normalized[i] = 0.0
        return normalized


# ── MODEL ────────────────────────────────────────────────────────────────────
class WildfireSpreadModel(nn.Module):
    """
    TerraMind encoder (frozen) + lightweight decoder head for spread prediction.
    Encoder: TerraMind-small-TiM (pretrained, mostly frozen)
    Decoder: Simple convolutional head outputting per-pixel spread probability
    """

    def __init__(self, freeze_encoder=True):
        super().__init__()

        print("Loading TerraMind-small encoder...", flush=True)

        try:
            self.encoder = BACKBONE_REGISTRY.build(
                'terramind_v1_small',
                pretrained=True,
                modalities=['S2L2A'],
            )
            print(">> encoder build done", flush=True)
        except Exception as e:
            import traceback
            print(f"[ERROR] {e}", flush=True)
            traceback.print_exc()
            raise

        print(">> starting freeze loop", flush=True)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        print(">> freeze done", flush=True)

        encoder_dim = 384
        print(">> building decoder", flush=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        print(">> decoder built", flush=True)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        print(">> upsample built", flush=True)

    def forward(self, x):
        inputs = {'S2L2A': x}

        features = self.encoder(inputs)

        # Take last feature map
        if isinstance(features, (list, tuple)):
            feat = features[-1]
        else:
            feat = features

        # ── ADD THIS BLOCK ──
        # ViT returns patch tokens: (batch, num_patches, dim) → (batch, dim, H, W)
        if feat.dim() == 3:
            batch, num_patches, dim = feat.shape
            h = w = int(num_patches ** 0.5)  # 196 → 14x14
            feat = feat.permute(0, 2, 1).reshape(batch, dim, h, w)
        # ────────────────────

        spread_map = self.decoder(feat)
        spread_map = self.upsample(spread_map)

        return spread_map


# ── LOSS FUNCTION ────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross Entropy.
    Better than BCE alone for imbalanced masks (only 11% burned pixels).
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # BCE component
        bce_loss = self.bce(pred, target)

        # Dice component
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return bce_loss + dice_loss


# ── METRICS ─────────────────────────────────────────────────────────────────
def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ── TRAINING LOOP ────────────────────────────────────────────────────────────
def train():
    # Create output dirs
    Path(CONFIG['model_save_dir']).mkdir(exist_ok=True)
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)

    # ← Model FIRST, before any rioxarray file I/O
    print("\nInitializing model...")
    model = WildfireSpreadModel(freeze_encoder=CONFIG['freeze_encoder'])
    model = model.to(CONFIG['device'])
    print(">> model ready", flush=True)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = BurnScarsDataset(CONFIG['data_dir'], split="training", image_size=CONFIG['image_size'])
    val_dataset = BurnScarsDataset(CONFIG['data_dir'], split="validation", image_size=CONFIG['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'])

    # Only optimize decoder parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    criterion = DiceBCELoss()

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print("=" * 60)

    for epoch in range(CONFIG['num_epochs']):

        # ── TRAIN ──
        model.train()
        train_losses = []

        for batch_imgs, batch_masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]"):
            batch_imgs  = batch_imgs.to(CONFIG['device'])
            batch_masks = batch_masks.to(CONFIG['device'])

            optimizer.zero_grad()
            predictions = model(batch_imgs)
            loss = criterion(predictions, batch_masks)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATE ──
        model.eval()
        val_losses = []
        val_ious = []

        with torch.no_grad():
            for batch_imgs, batch_masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]  "):
                batch_imgs  = batch_imgs.to(CONFIG['device'])
                batch_masks = batch_masks.to(CONFIG['device'])

                predictions = model(batch_imgs)
                loss = criterion(predictions, batch_masks)

                val_losses.append(loss.item())
                val_ious.append(calculate_iou(predictions, batch_masks))

        # ── METRICS ──
        avg_train_loss = np.mean(train_losses)
        avg_val_loss   = np.mean(val_losses)
        avg_val_iou    = np.mean(val_ious)
        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)

        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # ── SAVE BEST MODEL ──
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_path = Path(CONFIG['model_save_dir']) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': CONFIG,
            }, save_path)
            print(f"  ✓ New best model saved! IoU: {best_iou:.4f}")

    # ── PLOT TRAINING CURVES ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['val_iou'], color='green', label='Val IoU')
    ax2.axhline(y=best_iou, color='red', linestyle='--', label=f'Best IoU: {best_iou:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU Over Training')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(Path(CONFIG['output_dir']) / "training_curves.png", dpi=150)
    print(f"\nTraining curves saved to outputs/training_curves.png")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Best model saved to: {CONFIG['model_save_dir']}/best_model.pth")


# Bottom of model_a.py — replace the existing if __name__ block
if __name__ == "__main__":
    import traceback, sys

    try:
        train()
    except BaseException as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)