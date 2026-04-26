# infer.py
# AEON — On-Orbit Wildfire Intelligence Pipeline
# Usage:
#   python infer.py --input path/to/tile.tif
#   python infer.py --input path/to/tile.tif --output outputs/ --sample_idx 0
#   python infer.py --demo   (runs on built-in sample from validation set)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'      # silences the HF auth warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'          # suppresses a common HF side-warning

import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
import json

# ── PATHS — adjust if your models are elsewhere ──────────────────────────────
MODEL_A_PATH  = "models/best_model.pth"
MODEL_B_PATH  = "models/best_wildfire_model.pth"
MEANS_PATH    = "models/wildfire_means.npy"
STDS_PATH     = "models/wildfire_stds.npy"
DATA_DIR      = "data/hls_burn_scars"   # only needed for --demo mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── ENSEMBLE WEIGHTS (from training) ─────────────────────────────────────────
WEIGHT_A = 0.85   # TerraMind  (IoU 0.67)
WEIGHT_B = 0.15   # WildfireCNN (IoU 0.20)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS (inline so infer.py is self-contained)
# ─────────────────────────────────────────────────────────────────────────────
from terratorch import BACKBONE_REGISTRY

class WildfireSpreadModel(torch.nn.Module):
    """Model A: TerraMind encoder + decoder for HLS satellite imagery."""
    def __init__(self):
        super().__init__()
        import torch.nn as nn
        self.encoder = BACKBONE_REGISTRY.build(
            'terramind_v1_small', pretrained=True, modalities=['S2L2A']
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        encoder_dim = 384
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),         nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=1),         nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  1,   1), nn.Sigmoid()
        )
        self.upsample = nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)

    def forward(self, x):
        features = self.encoder({'S2L2A': x})
        feat = features[-1] if isinstance(features, (list,tuple)) else features
        if feat.dim() == 3:
            b, n, d = feat.shape
            h = w = int(n**0.5)
            feat = feat.permute(0,2,1).reshape(b,d,h,w)
        return self.upsample(self.decoder(feat))


class WildfireCNN(torch.nn.Module):
    """Model B: U-Net CNN for weather/terrain wildfire spread."""
    def __init__(self, in_channels=12):
        super().__init__()
        import torch.nn as nn

        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        self.enc1 = cb(in_channels, 64)
        self.enc2 = cb(64, 128)
        self.enc3 = cb(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = cb(256, 512)
        self.up3  = nn.ConvTranspose2d(512,256,2,stride=2); self.dec3 = cb(512,256)
        self.up2  = nn.ConvTranspose2d(256,128,2,stride=2); self.dec2 = cb(256,128)
        self.up1  = nn.ConvTranspose2d(128,64, 2,stride=2); self.dec1 = cb(128,64)
        self.out  = nn.Conv2d(64,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3),e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2),e1], 1))
        return self.sigmoid(self.out(d1))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    print(f"Device: {DEVICE}")
    print("Loading Model A (TerraMind)...")
    model_a = WildfireSpreadModel()
    ckpt_a  = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=False)
    model_a.load_state_dict(ckpt_a['model_state_dict'])
    model_a = model_a.to(DEVICE).eval()
    print(f"  Model A loaded. Training IoU: {ckpt_a['best_iou']:.4f}")

    print("Loading Model B (WildfireCNN)...")
    model_b = WildfireCNN(in_channels=12)
    ckpt_b  = torch.load(MODEL_B_PATH, map_location=DEVICE, weights_only=False)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b = model_b.to(DEVICE).eval()
    means   = ckpt_b['means']
    stds    = ckpt_b['stds']
    print(f"  Model B loaded. Training IoU: {ckpt_b['best_iou']:.4f}")

    return model_a, model_b, means, stds


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS INPUT
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_tif(tile_path: str):
    """Load a GeoTIFF (6-band HLS Sentinel-2) and prepare for inference."""
    import rioxarray as rxr
    tile = rxr.open_rasterio(tile_path).values.astype(np.float32)

    # Resize to 224×224
    _, h, w = tile.shape
    size = 224
    if h >= size and w >= size:
        sh, sw = (h-size)//2, (w-size)//2
        tile = tile[:, sh:sh+size, sw:sw+size]
    else:
        pad = np.zeros((tile.shape[0], size, size), dtype=np.float32)
        pad[:, :h, :w] = tile
        tile = pad

    # Normalize per band
    norm = np.zeros_like(tile)
    for i in range(tile.shape[0]):
        b = tile[i]
        bmin, bmax = b.min(), b.max()
        norm[i] = (b - bmin) / (bmax - bmin) if bmax > bmin else 0.0

    # Pad 6 HLS bands → 12 S2L2A bands
    img_12 = np.zeros((12, size, size), dtype=np.float32)
    img_12[1]  = norm[0]   # B02 Blue
    img_12[2]  = norm[1]   # B03 Green
    img_12[3]  = norm[2]   # B04 Red
    img_12[8]  = norm[3]   # B8A NIR
    img_12[10] = norm[4]   # B11 SWIR1
    img_12[11] = norm[5]   # B12 SWIR2

    return torch.tensor(img_12, dtype=torch.float32), norm


def preprocess_demo(sample_idx=0):
    """Load a sample from the HLS validation set (demo mode)."""
    import rioxarray as rxr
    val_dir   = Path(DATA_DIR) / "validation"
    all_tifs  = list(val_dir.glob("*.tif"))
    img_files = sorted([f for f in all_tifs if "_merged" in f.name])
    if not img_files:
        raise FileNotFoundError(f"No HLS validation tiles found in {val_dir}")
    path = img_files[sample_idx % len(img_files)]
    print(f"Demo mode: using {path.name}")
    return preprocess_tif(str(path))


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(img_tensor, model_a, model_b, means, stds):
    resize  = nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
    means_t = torch.tensor(means, dtype=torch.float32).view(1,-1,1,1).to(DEVICE)
    stds_t  = torch.tensor(stds,  dtype=torch.float32).view(1,-1,1,1).to(DEVICE)

    img_in = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_a = model_a(img_in)

        proxy = torch.zeros(1,12,64,64, device=DEVICE)
        proxy[:,2]  = nn.functional.adaptive_avg_pool2d(img_in[:,8:9], (64,64)).squeeze(1)
        proxy[:,9]  = nn.functional.adaptive_avg_pool2d(img_in[:,10:11],(64,64)).squeeze(1)
        proxy_norm  = (proxy - means_t[:,:,:64,:64]) / stds_t[:,:,:64,:64]
        pred_b      = resize(model_b(proxy_norm))

        ensemble = WEIGHT_A * pred_a + WEIGHT_B * pred_b

    return (pred_a.squeeze().cpu().numpy(),
            pred_b.squeeze().cpu().numpy(),
            ensemble.squeeze().cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE + REPORT
# ─────────────────────────────────────────────────────────────────────────────
def analyze(risk_map, sample_idx=0):
    H, W = risk_map.shape
    HIGH   = risk_map > 0.65
    MEDIUM = (risk_map > 0.35) & (risk_map <= 0.65)

    total_px   = H * W
    high_pct   = HIGH.sum()   / total_px * 100
    medium_pct = MEDIUM.sum() / total_px * 100
    high_km2   = HIGH.sum()   * 100 / 1e6
    medium_km2 = MEDIUM.sum() * 100 / 1e6

    quadrants = {
        "Northwest": risk_map[:H//2, :W//2],
        "Northeast": risk_map[:H//2, W//2:],
        "Southwest": risk_map[H//2:, :W//2],
        "Southeast": risk_map[H//2:, W//2:],
    }
    hottest_quad = max(quadrants, key=lambda k: quadrants[k].mean())

    directional = {
        "North": risk_map[:H//3, :].mean(),
        "South": risk_map[2*H//3:, :].mean(),
        "East":  risk_map[:, 2*W//3:].mean(),
        "West":  risk_map[:, :W//3].mean(),
    }
    primary   = max(directional, key=directional.get)
    secondary = sorted(directional, key=directional.get, reverse=True)[1]

    if high_pct > 20:   severity = "CRITICAL"
    elif high_pct > 10: severity = "HIGH"
    elif high_pct > 3:  severity = "MODERATE"
    elif high_pct > 0.5:severity = "LOW"
    else:               severity = "MINIMAL"

    if HIGH.sum() > 0:
        hr = np.where(HIGH.any(axis=1))[0]
        hc = np.where(HIGH.any(axis=0))[0]
        sh = hr.max()-hr.min() if len(hr)>1 else 0
        sv = hc.max()-hc.min() if len(hc)>1 else 0
        pattern = "dispersed across a wide front" if (sh>H//3 or sv>W//3) else "concentrated in a localized zone"
    else:
        pattern = "no significant fire activity detected"

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    report = {
        "timestamp": ts, "satellite": "AEON-1", "sample_id": sample_idx,
        "severity": severity,
        "high_risk_pct": round(high_pct,2), "medium_risk_pct": round(medium_pct,2),
        "high_risk_km2": round(high_km2,2), "medium_risk_km2": round(medium_km2,2),
        "primary_spread_dir": primary, "secondary_spread_dir": secondary,
        "hottest_quadrant": hottest_quad, "fire_pattern": pattern,
        "directional_risk": {k: round(float(v),3) for k,v in directional.items()},
    }

    if severity in ("CRITICAL","HIGH"):
        rec = f"Deploy resources to {hottest_quad}. Contain {primary}ward spread immediately."
    elif severity == "MODERATE":
        rec = f"Increase surveillance. Alert {primary} sector crews."
    else:
        rec = "Routine monitoring. No immediate action required."

    text = f"""
╔══════════════════════════════════════════════════════════╗
║           AEON WILDFIRE SPREAD ALERT                     ║
║           On-Orbit Intelligence Report                   ║
╚══════════════════════════════════════════════════════════╝
Timestamp  : {ts}
Satellite  : AEON-1
Severity   : {severity}

FIRE SPREAD ASSESSMENT
  High-risk zone   : {high_pct:.1f}%  ({high_km2:.2f} km²)
  Medium-risk zone : {medium_pct:.1f}%  ({medium_km2:.2f} km²)

SPREAD DIRECTION
  Primary   : {primary}ward   (index: {directional[primary]:.3f})
  Secondary : {secondary}ward  (index: {directional[secondary]:.3f})
  Pattern   : Fire is {pattern}

DIRECTIONAL RISK
  North : {'█'*int(directional['North']*20):<20} {directional['North']:.3f}
  South : {'█'*int(directional['South']*20):<20} {directional['South']:.3f}
  East  : {'█'*int(directional['East']*20):<20}  {directional['East']:.3f}
  West  : {'█'*int(directional['West']*20):<20}  {directional['West']:.3f}

RECOMMENDATION: {rec}

TRANSMISSION SUMMARY
  Raw image  : ~500 MB  (NOT transmitted)
  Risk map   : ~0.2 MB  (transmitted)
  This report: ~2 KB    (transmitted)
  Saving     : ~2500x bandwidth reduction
═══════════════════════════════════════════════════════════
"""
    return report, text


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(img_tensor, norm_bands, pred_a, pred_b, ensemble,
                 report, text, output_dir, sample_idx):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20,10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)
    fig.patch.set_facecolor('#0a0a0a')

    axes = [fig.add_subplot(gs[0,i]) for i in range(4)]
    ax_bar = fig.add_subplot(gs[1,0])
    ax_dir = fig.add_subplot(gs[1,1])
    ax_txt = fig.add_subplot(gs[1,2:])
    for ax in axes+[ax_bar,ax_dir,ax_txt]:
        ax.set_facecolor('#0a0a0a')
        for s in ax.spines.values(): s.set_edgecolor('#333')

    tk = dict(color='white', fontsize=10, fontweight='bold')

    # Row 1
    rgb = np.clip(norm_bands[[2,1,0]].transpose(1,2,0), 0, 1)
    axes[0].imshow(rgb);                                       axes[0].set_title('Satellite Input (RGB)', **tk); axes[0].axis('off')
    im1=axes[1].imshow(pred_a, cmap='RdYlGn_r', vmin=0,vmax=1); axes[1].set_title('Model A — TerraMind\n(Spectral)', **tk); axes[1].axis('off')
    im2=axes[2].imshow(pred_b, cmap='RdYlGn_r', vmin=0,vmax=1); axes[2].set_title('Model B — WildfireCNN\n(Weather/Terrain)', **tk); axes[2].axis('off')
    sev_color = {'CRITICAL':'#ff2222','HIGH':'#ff6622','MODERATE':'#ffaa00','LOW':'#aadd00','MINIMAL':'#44cc44'}.get(report['severity'],'white')
    im3=axes[3].imshow(ensemble, cmap='RdYlGn_r', vmin=0,vmax=1); axes[3].set_title('Ensemble Risk Map', **tk)
    axes[3].set_xlabel(f"Severity: {report['severity']}", color=sev_color, fontsize=11, fontweight='bold'); axes[3].axis('off')
    for im,ax in zip([im1,im2,im3],axes[1:]):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white',labelcolor='white')

    # Bar: risk zones
    zones  = ['High','Medium','Low']
    vals   = [report['high_risk_pct'], report['medium_risk_pct'],
              100-report['high_risk_pct']-report['medium_risk_pct']]
    colors = ['#ff3333','#ffaa00','#44cc44']
    bars   = ax_bar.bar(zones, vals, color=colors, edgecolor='#333')
    ax_bar.set_title('Risk Zone Breakdown', **tk); ax_bar.set_ylabel('%', color='white')
    ax_bar.tick_params(colors='white'); ax_bar.set_facecolor('#0a0a0a')
    for bar,v in zip(bars,vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}%',
                    ha='center', color='white', fontsize=9)

    # Directional bars
    dirs  = ['North','South','East','West']
    drisk = [report['directional_risk'][d] for d in dirs]
    dcols = ['#ff4444' if v>0.5 else '#ffaa00' if v>0.25 else '#44cc44' for v in drisk]
    hb    = ax_dir.barh(dirs, drisk, color=dcols, edgecolor='#333')
    ax_dir.set_title('Directional Spread Risk', **tk); ax_dir.set_xlim(0,1)
    ax_dir.tick_params(colors='white'); ax_dir.set_facecolor('#0a0a0a')
    for bar,v in zip(hb,drisk):
        ax_dir.text(v+0.01, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', color='white', fontsize=9)

    # Report text
    ax_txt.axis('off')
    short = (
        f"AEON-1 ON-ORBIT REPORT  |  {report['timestamp']}\n"
        f"{'─'*52}\n"
        f"SEVERITY        : {report['severity']}\n"
        f"High-risk area  : {report['high_risk_pct']:.1f}%  ({report['high_risk_km2']:.2f} km²)\n"
        f"Medium-risk area: {report['medium_risk_pct']:.1f}%  ({report['medium_risk_km2']:.2f} km²)\n\n"
        f"Primary spread  : {report['primary_spread_dir']}ward\n"
        f"Secondary spread: {report['secondary_spread_dir']}ward\n"
        f"Hottest sector  : {report['hottest_quadrant']}\n"
        f"Pattern         : {report['fire_pattern']}\n\n"
        f"{'─'*52}\n"
        f"RECOMMENDATION:\n"
    )
    if report['severity'] in ('CRITICAL','HIGH'):
        short += f"⚠  Deploy to {report['hottest_quadrant']}. Contain {report['primary_spread_dir']}ward spread."
    elif report['severity'] == 'MODERATE':
        short += f"⚡ Increase surveillance. Alert {report['primary_spread_dir']} crews."
    else:
        short += "✓  Routine monitoring. No action required."
    short += f"\n\n{'─'*52}\nPayload: ~0.2MB map + ~2KB report  (saved ~500MB)"

    ax_txt.text(0.02, 0.98, short, transform=ax_txt.transAxes, fontsize=9,
                va='top', fontfamily='monospace', color='#00ff88',
                bbox=dict(boxstyle='round', facecolor='#111', edgecolor='#00ff88', alpha=0.8))

    fig.suptitle('AEON — On-Orbit Wildfire Intelligence Pipeline',
                 color='white', fontsize=14, fontweight='bold')

    png_path  = out / f"onorbit_report_{sample_idx:03d}.png"
    txt_path  = out / f"onorbit_report_{sample_idx:03d}.txt"
    json_path = out / f"onorbit_report_{sample_idx:03d}.json"

    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    txt_path.write_text(text, encoding='utf-8')
    json_path.write_text(json.dumps(report, indent=2))

    print(f"\nOutputs saved:")
    print(f"  {png_path}   ← risk map visualization")
    print(f"  {txt_path}   ← text alert")
    print(f"  {json_path}  ← machine-readable report")
    print(f"\n  Total transmitted payload: ~0.2 MB  (vs ~500 MB raw)")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEON On-Orbit Wildfire Inference")
    parser.add_argument("--input",      type=str, help="Path to input GeoTIFF (.tif)")
    parser.add_argument("--output",     type=str, default="outputs", help="Output directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index (for naming outputs)")
    parser.add_argument("--demo",       action="store_true", help="Run on built-in validation sample")
    args = parser.parse_args()

    if not args.input and not args.demo:
        print("Usage:")
        print("  python infer.py --input path/to/tile.tif")
        print("  python infer.py --demo")
        sys.exit(1)

    # Load models
    model_a, model_b, means, stds = load_models()

    # Load input
    if args.demo:
        img_tensor, norm_bands = preprocess_demo(args.sample_idx)
    else:
        img_tensor, norm_bands = preprocess_tif(args.input)

    # Run inference
    print("Running ensemble inference...")
    pred_a, pred_b, ensemble = run_inference(img_tensor, model_a, model_b, means, stds)

    # Analyze
    report, text = analyze(ensemble, args.sample_idx)
    print(text)

    # Save
    save_outputs(img_tensor, norm_bands, pred_a, pred_b, ensemble,
                 report, text, args.output, args.sample_idx)