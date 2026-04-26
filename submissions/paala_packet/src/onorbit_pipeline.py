# 9_onorbit_pipeline.py
# Full on-orbit inference pipeline:
# Satellite image → Ensemble → Risk map + Natural language report
# Everything computed in space, only results sent to Earth

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
import importlib
import json

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from src.model_a import WildfireSpreadModel, BurnScarsDataset
from src.model_b import WildfireCNN
from configs.model_config import CONFIG_A, CONFIG_B

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
def load_models():
    # Model A
    ckpt_a = torch.load("models/best_model.pth", map_location=DEVICE, weights_only=False)
    model_a = WildfireSpreadModel(freeze_encoder=True)
    model_a.load_state_dict(ckpt_a['model_state_dict'])
    model_a = model_a.to(DEVICE).eval()

    # Model B
    ckpt_b = torch.load("models/best_wildfire_model.pth", map_location=DEVICE, weights_only=False)
    model_b = WildfireCNN(in_channels=12)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b = model_b.to(DEVICE).eval()
    means = ckpt_b['means']
    stds = ckpt_b['stds']

    print("Both models loaded on-orbit.")
    return model_a, model_b, means, stds


# ── ENSEMBLE INFERENCE ────────────────────────────────────────────────────────
def run_ensemble(img_tensor, model_a, model_b, means, stds, wa=0.85, wb=0.15):
    """
    Run full ensemble inference on a single satellite tile.
    wa/wb: weights for Model A and B (found from onorbit_pipeline.py)
    """
    resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
    means_t = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1).to(DEVICE)
    stds_t = torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1).to(DEVICE)

    img_input = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Model A — satellite spectral bands
        pred_a = model_a(img_input)  # (1,1,224,224)

        # Model B — weather/terrain proxy from satellite bands
        # (In real deployment, these come from onboard meteorological sensors)
        proxy = torch.zeros(1, 12, 64, 64, device=DEVICE)
        proxy[:, 2] = nn.functional.adaptive_avg_pool2d(
            img_input[:, 8:9], (64, 64)).squeeze(1)  # NDVI ≈ NIR
        proxy[:, 9] = nn.functional.adaptive_avg_pool2d(
            img_input[:, 10:11], (64, 64)).squeeze(1)  # ERC ≈ SWIR1
        proxy_norm = (proxy - means_t[:, :, :64, :64]) / stds_t[:, :, :64, :64]
        pred_b = resize(model_b(proxy_norm))  # (1,1,224,224)

        # Ensemble
        ensemble = wa * pred_a + wb * pred_b

    return (pred_a.squeeze().cpu().numpy(),
            pred_b.squeeze().cpu().numpy(),
            ensemble.squeeze().cpu().numpy())


# ── ANALYZE RISK MAP → NATURAL LANGUAGE ───────────────────────────────────────
def analyze_risk_map(risk_map, sample_idx=0):
    """
    Analyze the ensemble risk map and generate a structured report.
    This runs entirely on the satellite — only the report is sent to Earth.

    Assumes north=up, east=right in the 224x224 grid.
    """
    H, W = risk_map.shape  # 224, 224

    # ── THRESHOLDS ──
    HIGH = risk_map > 0.65
    MEDIUM = (risk_map > 0.35) & (risk_map <= 0.65)
    LOW = risk_map <= 0.35

    total_px = H * W
    high_pct = HIGH.sum() / total_px * 100
    medium_pct = MEDIUM.sum() / total_px * 100
    high_km2 = HIGH.sum() * 100 / 1e6  # each pixel ≈ 10m×10m
    medium_km2 = MEDIUM.sum() * 100 / 1e6

    # ── SPATIAL ANALYSIS: which quadrant has most fire? ──
    quadrants = {
        "Northwest": risk_map[:H // 2, :W // 2],
        "Northeast": risk_map[:H // 2, W // 2:],
        "Southwest": risk_map[H // 2:, :W // 2],
        "Southeast": risk_map[H // 2:, W // 2:],
    }
    quad_means = {k: v.mean() for k, v in quadrants.items()}
    hottest_quad = max(quad_means, key=quad_means.get)

    # ── DIRECTIONAL SPREAD: compare top/bottom/left/right thirds ──
    thirds_h = {
        "North": risk_map[:H // 3, :],
        "Center": risk_map[H // 3:2 * H // 3, :],
        "South": risk_map[2 * H // 3:, :],
    }
    thirds_v = {
        "West": risk_map[:, :W // 3],
        "Center": risk_map[:, W // 3:2 * W // 3],
        "East": risk_map[:, 2 * W // 3:],
    }

    north_risk = thirds_h["North"].mean()
    south_risk = thirds_h["South"].mean()
    east_risk = thirds_v["East"].mean()
    west_risk = thirds_v["West"].mean()
    center_risk = risk_map[H // 3:2 * H // 3, W // 3:2 * W // 3].mean()

    # Primary spread direction
    directional = {
        "North": north_risk, "South": south_risk,
        "East": east_risk, "West": west_risk,
    }
    primary_direction = max(directional, key=directional.get)
    secondary_direction = sorted(directional, key=directional.get, reverse=True)[1]

    # ── SEVERITY LEVEL ──
    if high_pct > 20:
        severity = "CRITICAL"
    elif high_pct > 10:
        severity = "HIGH"
    elif high_pct > 3:
        severity = "MODERATE"
    elif high_pct > 0.5:
        severity = "LOW"
    else:
        severity = "MINIMAL"

    # ── TREND: is fire concentrated or dispersed? ──
    if HIGH.sum() > 0:
        high_rows = np.where(HIGH.any(axis=1))[0]
        high_cols = np.where(HIGH.any(axis=0))[0]
        spread_h = high_rows.max() - high_rows.min() if len(high_rows) > 1 else 0
        spread_v = high_cols.max() - high_cols.min() if len(high_cols) > 1 else 0
        pattern = "dispersed across a wide front" if (
                    spread_h > H // 3 or spread_v > W // 3) else "concentrated in a localized zone"
    else:
        pattern = "no significant fire activity detected"

    # ── GENERATE REPORT ──
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    report = {
        "timestamp": timestamp,
        "satellite": "AEON-1",
        "sample_id": sample_idx,
        "severity": severity,
        "high_risk_pct": round(high_pct, 2),
        "medium_risk_pct": round(medium_pct, 2),
        "high_risk_km2": round(high_km2, 2),
        "medium_risk_km2": round(medium_km2, 2),
        "primary_spread_dir": primary_direction,
        "secondary_spread_dir": secondary_direction,
        "hottest_quadrant": hottest_quad,
        "fire_pattern": pattern,
        "center_risk": round(float(center_risk), 3),
        "directional_risk": {k: round(float(v), 3) for k, v in directional.items()},
    }

    # ── HUMAN-READABLE TEXT (what gets transmitted to Earth) ──
    text = f"""
╔══════════════════════════════════════════════════════════╗
║           AEON WILDFIRE SPREAD ALERT                     ║
║           On-Orbit Intelligence Report                   ║
╚══════════════════════════════════════════════════════════╝

Timestamp : {timestamp}
Satellite : AEON-1
Severity  : {severity}

FIRE SPREAD ASSESSMENT
──────────────────────────────────────────────────────────
High-risk zone   : {high_pct:.1f}% of observed tile ({high_km2:.2f} km²)
Medium-risk zone : {medium_pct:.1f}% of observed tile ({medium_km2:.2f} km²)

SPREAD DIRECTION
──────────────────────────────────────────────────────────
Primary spread   : {primary_direction}ward  (risk index: {directional[primary_direction]:.3f})
Secondary spread : {secondary_direction}ward (risk index: {directional[secondary_direction]:.3f})
Hottest quadrant : {hottest_quad}
Pattern          : Fire is {pattern}

DIRECTIONAL RISK INDEX (0=safe, 1=critical)
  North : {'█' * int(north_risk * 20):<20} {north_risk:.3f}
  South : {'█' * int(south_risk * 20):<20} {south_risk:.3f}
  East  : {'█' * int(east_risk * 20):<20}  {east_risk:.3f}
  West  : {'█' * int(west_risk * 20):<20}  {west_risk:.3f}

RECOMMENDATION
──────────────────────────────────────────────────────────"""

    if severity == "CRITICAL":
        text += f"\n⚠️  IMMEDIATE ACTION REQUIRED"
        text += f"\n    Deploy aerial resources to {hottest_quad} sector."
        text += f"\n    Evacuate zones in {primary_direction} and {secondary_direction} corridors."
        text += f"\n    Establish firebreak {secondary_direction} of current perimeter."
    elif severity == "HIGH":
        text += f"\n⚠️  HIGH PRIORITY RESPONSE"
        text += f"\n    Pre-position resources near {hottest_quad} sector."
        text += f"\n    Monitor {primary_direction}ward spread — containment recommended."
    elif severity == "MODERATE":
        text += f"\n⚡ ELEVATED MONITORING"
        text += f"\n    Increase aerial surveillance frequency."
        text += f"\n    Alert ground crews in {primary_direction} sector."
    else:
        text += f"\n✓  ROUTINE MONITORING — No immediate action required."

    text += f"""

DATA TRANSMISSION SUMMARY
──────────────────────────────────────────────────────────
Raw image size   : ~500 MB (NOT transmitted)
Risk map payload : ~0.2 MB (transmitted)
This report      : ~2 KB  (transmitted)
Bandwidth saving : ~2500×
══════════════════════════════════════════════════════════
"""
    return report, text


# ── GENERATE FINAL OUTPUT IMAGE + REPORT ──────────────────────────────────────
def generate_output(img_tensor, pred_a, pred_b, ensemble,
                    report, text, sample_idx, mask_tensor=None):
    """
    Generate the final output that gets transmitted to Earth:
    1. Risk map image (0.2 MB)
    2. Text report (2 KB)
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ── ROW 1: Images ──
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_model_a = fig.add_subplot(gs[0, 1])
    ax_model_b = fig.add_subplot(gs[0, 2])
    ax_ensemble = fig.add_subplot(gs[0, 3])

    # ── ROW 2: Risk breakdown + report text ──
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_dir = fig.add_subplot(gs[1, 1])
    ax_report = fig.add_subplot(gs[1, 2:])

    fig.patch.set_facecolor('#0a0a0a')
    for ax in [ax_rgb, ax_model_a, ax_model_b, ax_ensemble,
               ax_bar, ax_dir, ax_report]:
        ax.set_facecolor('#0a0a0a')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    title_kwargs = dict(color='white', fontsize=11, fontweight='bold', pad=8)
    label_kwargs = dict(color='#aaaaaa', fontsize=8)

    # RGB
    rgb = img_tensor.numpy()[[3, 2, 1]].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title('Satellite Input', **title_kwargs)
    ax_rgb.axis('off')

    # Model A
    im_a = ax_model_a.imshow(pred_a, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax_model_a.set_title('Model A\n(TerraMind — Spectral)', **title_kwargs)
    ax_model_a.axis('off')
    plt.colorbar(im_a, ax=ax_model_a, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white',
                                                                                         labelcolor='white')

    # Model B
    im_b = ax_model_b.imshow(pred_b, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax_model_b.set_title('Model B\n(Weather CNN — Environmental)', **title_kwargs)
    ax_model_b.axis('off')
    plt.colorbar(im_b, ax=ax_model_b, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white',
                                                                                         labelcolor='white')

    # Ensemble
    im_e = ax_ensemble.imshow(ensemble, cmap='RdYlGn_r', vmin=0, vmax=1)
    sev = report['severity']
    sev_color = {'CRITICAL': '#ff2222', 'HIGH': '#ff6622', 'MODERATE': '#ffaa00',
                 'LOW': '#aadd00', 'MINIMAL': '#44cc44'}.get(sev, 'white')
    ax_ensemble.set_title(f'Ensemble Risk Map\n', **title_kwargs)
    ax_ensemble.set_xlabel(f'Severity: {sev}', color=sev_color, fontsize=11, fontweight='bold')
    ax_ensemble.axis('off')
    plt.colorbar(im_e, ax=ax_ensemble, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white',
                                                                                          labelcolor='white')

    # Bar chart: risk zone breakdown
    zones = ['High Risk', 'Medium Risk', 'Low Risk']
    values = [report['high_risk_pct'], report['medium_risk_pct'],
              100 - report['high_risk_pct'] - report['medium_risk_pct']]
    colors = ['#ff3333', '#ffaa00', '#44cc44']
    bars = ax_bar.bar(zones, values, color=colors, edgecolor='#333333')
    ax_bar.set_title('Risk Zone Breakdown (%)', **title_kwargs)
    ax_bar.set_ylabel('% of Tile', color='white')
    ax_bar.tick_params(colors='white')
    ax_bar.set_facecolor('#0a0a0a')
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9)

    # Directional risk compass-style bar
    dirs = ['North', 'South', 'East', 'West']
    drisk = [report['directional_risk'][d] for d in dirs]
    dcolors = ['#ff4444' if v > 0.5 else '#ffaa00' if v > 0.25 else '#44cc44' for v in drisk]
    hbars = ax_dir.barh(dirs, drisk, color=dcolors, edgecolor='#333333')
    ax_dir.set_title('Directional Spread Risk', **title_kwargs)
    ax_dir.set_xlim(0, 1)
    ax_dir.tick_params(colors='white')
    ax_dir.set_facecolor('#0a0a0a')
    ax_dir.axvline(0.5, color='#ff4444', linestyle='--', alpha=0.5, linewidth=1)
    ax_dir.axvline(0.25, color='#ffaa00', linestyle='--', alpha=0.5, linewidth=1)
    for bar, val in zip(hbars, drisk):
        ax_dir.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', color='white', fontsize=9)

    # Report text panel
    ax_report.axis('off')
    short_report = (
        f"AEON-1 ON-ORBIT REPORT  |  {report['timestamp']}\n"
        f"{'─' * 55}\n"
        f"SEVERITY: {report['severity']}\n\n"
        f"High-risk area  :  {report['high_risk_pct']:.1f}%  ({report['high_risk_km2']:.2f} km²)\n"
        f"Medium-risk area:  {report['medium_risk_pct']:.1f}%  ({report['medium_risk_km2']:.2f} km²)\n\n"
        f"Primary spread  :  {report['primary_spread_dir']}ward\n"
        f"Secondary spread:  {report['secondary_spread_dir']}ward\n"
        f"Hottest sector  :  {report['hottest_quadrant']}\n"
        f"Fire pattern    :  {report['fire_pattern']}\n\n"
        f"{'─' * 55}\n"
        f"RECOMMENDATION:\n"
    )
    if report['severity'] in ('CRITICAL', 'HIGH'):
        short_report += f"⚠ Deploy resources to {report['hottest_quadrant']} sector.\n"
        short_report += f"  Contain {report['primary_spread_dir']}ward spread immediately."
    elif report['severity'] == 'MODERATE':
        short_report += f"⚡ Increase surveillance. Alert {report['primary_spread_dir']} sector crews."
    else:
        short_report += "✓ Routine monitoring. No immediate action required."

    short_report += f"\n\n{'─' * 55}\n"
    short_report += f"Payload: ~0.2MB map + ~2KB report  (saved ~500MB)"

    ax_report.text(0.02, 0.98, short_report, transform=ax_report.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   color='#00ff88', bbox=dict(boxstyle='round', facecolor='#111111',
                                              edgecolor='#00ff88', alpha=0.8))

    fig.suptitle('AEON — On-Orbit Wildfire Intelligence Pipeline',
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    out_path = f"outputs/onorbit_report_{sample_idx:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"Output image saved → {out_path}")
    plt.show()

    # Also save text report
    txt_path = f"outputs/onorbit_report_{sample_idx:03d}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text report saved  → {txt_path}")

    json_path = f"outputs/onorbit_report_{sample_idx:03d}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved  → {json_path}")

    return out_path, txt_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)

    model_a, model_b, means, stds = load_models()

    # Run on 3 validation samples
    val_ds = BurnScarsDataset(CONFIG_A['data_dir'], split="validation", image_size=224)

    SAMPLE_INDICES = [0, 25, 50]  # change these to any index you want

    for idx in SAMPLE_INDICES:
        print(f"\n{'=' * 60}")
        print(f"Processing sample {idx}...")

        img_tensor, mask_tensor = val_ds[idx]

        pred_a, pred_b, ensemble = run_ensemble(
            img_tensor, model_a, model_b, means, stds,
            wa=0.85, wb=0.15  # update wa/wb from 8_ensemble_pipeline output
        )

        report, text = analyze_risk_map(ensemble, sample_idx=idx)

        print(text)

        generate_output(
            img_tensor, pred_a, pred_b, ensemble,
            report, text, idx, mask_tensor
        )

    print("\nAll samples processed.")
    print("Files saved to outputs/")
    print("These are the ONLY files that would be transmitted from orbit to Earth.")