# AEON — On-Orbit Wildfire Intelligence

**Team:** PAALA PACKET
**Track:** Earth Observation / On-Orbit AI
**Models:** TerraMind-v1-small (fine-tuned) + WildfireCNN (trained from scratch)

## Download Model Weights
Models are hosted on Google Drive due to GitHub size limits.

[Download models folder](https://drive.google.com/drive/folders/1hXqQSNBEMCne4Jq7Q2qPjIduc4DHUy3_?usp=sharing)

Place downloaded files in:
models/
├── best_model.pth
├── best_wildfire_model.pth
├── wildfire_means.npy
└── wildfire_stds.npy

---

## 1. What Problem Are We Solving?

Wildfires are accelerating in scale and speed, and the bottleneck in early response is not sensor coverage — it is time. Satellites already image active fire zones, but raw imagery must be downlinked to Earth (~500 MB per tile), queued for ground processing, and analyzed before any alert reaches firefighting crews. That pipeline takes hours. A fire moving at 10 km/h does not wait. Our customer is any agency or government that operates wildfire response — national forest services, disaster management authorities, or defense agencies with airborne firefighting assets. They would pay for guaranteed sub-minute alerts with actionable spread direction, because earlier deployment of aerial resources directly reduces area burned and lives lost.

---

## 2. What Did We Build?

We built a two-model ensemble inference pipeline designed to run entirely on-orbit, transmitting only a compressed risk map and a natural language alert to Earth instead of raw imagery.

**Model A — TerraMind (Spectral):**
Fine-tuned `terramind_v1_small` (IBM/NASA pretrained ViT encoder) on the HLS Burn Scars dataset (7,607 Sentinel-2 tiles, 6 spectral bands). The encoder is frozen; only a lightweight 4-layer CNN decoder is trained to output a per-pixel burn probability map at 224×224 resolution. Fine-tuning uses DiceBCE loss with a cosine LR schedule over 20 epochs.

**Model B — WildfireCNN (Environmental):**
A 7.7M-parameter U-Net trained from scratch on the Next Day Wildfire Spread dataset (14,979 samples). Inputs are 12 environmental channels — elevation, PDSI drought index, NDVI, precipitation, humidity, wind speed/direction, min/max temperature, energy release component, population density, and previous fire mask. Trained with Focal+Dice loss and weighted sampling to handle the extreme class imbalance (1.07% fire pixels).

**Ensemble + On-Orbit Report:**
Both models run on the satellite. Their outputs are blended (w=0.85 / 0.15) and analyzed spatially to produce directional spread assessment, severity classification, and a natural language recommendation — all transmitted as a 0.2 MB risk map + 2 KB text report.

---

## 3. How Did We Measure It?

| Model | Dataset | Metric | Score | Baseline |
|---|---|---|---|---|
| Model A (TerraMind) | HLS Burn Scars (val) | IoU | **0.67** | 0.00 (predict nothing) |
| Model B (WildfireCNN) | NDWS (eval split) | IoU | **0.21** | ~0.15 (published paper) |
| Ensemble | HLS Burn Scars (val) | IoU | **0.67** | — |

Baseline for Model A is a static predictor that outputs zeros (no spread). Model A improves over this baseline by +0.67 IoU. Model B matches and slightly exceeds the IoU reported in the original Next Day Wildfire Spread paper (~0.15–0.20) which establishes that weather/terrain alone is a weak but non-trivial predictor. The ensemble is dominated by Model A (satellite has higher signal), with Model B contributing environmental context that is unavailable from spectral bands alone.

---

## 4. The Orbital-Compute Story

| Component | Detail |
|---|---|
| Model A size | ~300 MB (TerraMind encoder frozen, decoder 2.1M params) |
| Model B size | ~30 MB (7.7M params, U-Net) |
| Total on-orbit footprint | ~330 MB |
| Inference time (RTX 3050) | < 1 second per tile |
| Raw image size | ~500 MB |
| Transmitted payload | ~0.2 MB risk map + ~2 KB report |
| Bandwidth reduction | **~2,500×** |

The TerraMind encoder is frozen during fine-tuning, meaning only the 2.1M-parameter decoder needs to be updated or re-uploaded for future model iterations — a 143× smaller update package than the full model. On real space hardware (e.g., Unibap iX10 or Ubotica CogniSAT), the frozen ViT encoder maps naturally to NPU acceleration. Model B's U-Net is well within the compute envelope of current edge AI chips (Nvidia Jetson class). The entire pipeline produces actionable output before the satellite completes its overpass, enabling real-time downlink of intelligence rather than data.

---

## 5. What Doesn't Work Yet

**Model B plateaus at IoU ~0.21.** Weather and terrain data alone are weak predictors of next-day fire extent. The 64×64 patch resolution captures local conditions but misses mesoscale wind patterns. A larger receptive field or temporal stacking (multiple days of weather) would likely push this higher.

**Model B inputs are proxied on-orbit.** In the current implementation, Model B's weather channels (wind, humidity, temperature) are approximated from HLS spectral bands when running on satellite. A real deployment would require the satellite to carry or receive meteorological sensor feeds — either onboard instruments or uplinked NWP model output.

**No georeferencing in the report.** The spread direction (North/South/East/West) is computed relative to image orientation, not true geographic north. Integration with the satellite's attitude control system would fix this.

**Single-tile inference only.** Large fires spanning multiple tiles are not stitched. A mosaicking step before inference would improve coverage for major fire events.

---

## Setup Instructions

**Requirements:** Python 3.11 or 3.12 (not 3.13), NVIDIA GPU recommended

```bash
# 1. Create and activate virtual environment
py -3.11 -m venv venv_new
.\venv_new\Scripts\activate          # Windows
# source venv_new/bin/activate       # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. If GPU not detected or CUDA errors, install PyTorch with CUDA explicitly:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# 4. Download HLS Burn Scars dataset (~2 GB)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ibm-nasa-geospatial/hls_burn_scars',
    repo_type='dataset',
    local_dir='./data/hls_burn_scars'
)"

# 5. Download Next Day Wildfire Spread dataset
# https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
# Place .tfrecord files in: data/next_day_wildfire/
```

---

## Running Inference (Judges: Start Here)

```bash
# Demo mode — uses built-in validation sample, no extra data needed
python infer.py --demo

# Run on your own satellite tile (.tif)
python infer.py --input sample_input/sample_tile.tif

# Specify output directory
python infer.py --input sample_input/sample_tile.tif --output outputs/
```

**Output files (what would be transmitted from orbit to Earth):**
```
outputs/onorbit_report_000.png    ~0.2 MB   risk map visualization
outputs/onorbit_report_000.txt    ~2 KB     human-readable alert
outputs/onorbit_report_000.json   ~1 KB     machine-readable for GIS
```

---

## Retraining From Scratch

```bash
# Train Model A (TerraMind on HLS data) — ~2 hours on RTX 3050
python train.py --model a

# Train Model B (WildfireCNN on NDWS data) — ~45 min on RTX 3050
python train.py --model b
```

---
