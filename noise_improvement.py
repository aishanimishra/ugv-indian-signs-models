"""
YOLO26n — Noise Robustness Improvement Algorithm
=================================================
Compares 3 denoising methods × 2 threshold strategies vs baseline (no improvement)

Preprocessing methods:
    1. Gaussian Blur
    2. Median Filter
    3. Non-Local Means (NLM)

Post-processing threshold strategies:
    1. Fixed lower threshold per sigma level
    2. Auto-adapt threshold based on estimated noise

For each combination:
    - Runs all 591 images at each sigma level
    - Records detection rate %, mean conf, max conf, degradation %
    - Saves full comparison table as CSV + Excel
    - Generates comparison plots

Run:
    python noise_improvement.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(BASE, "valid", "images")
OUT_DIR    = os.path.join(BASE, "noise_improvement_results")
MODEL_PATH = os.path.join(BASE, "yolo26n.onnx")

SIGMAS          = [0, 25, 50, 75, 100, 125, 150]
BASE_CONF       = 0.25      # Default confidence threshold

# Fixed thresholds per sigma level (lower = more lenient as noise increases)
FIXED_THRESH_MAP = {
    0:   0.25,
    25:  0.22,
    50:  0.18,
    75:  0.15,
    100: 0.12,
    125: 0.10,
    150: 0.08,
}

# ── Preprocessing methods ──────────────────────────────────────────────────────
def preprocess_none(image: np.ndarray) -> np.ndarray:
    return image.copy()

def preprocess_gaussian(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)

def preprocess_median(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, 5)

def preprocess_nlm(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None,
                                           h=10, hColor=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)

# ── Post-processing threshold strategies ──────────────────────────────────────
def threshold_fixed(sigma: int) -> float:
    """Fixed lower threshold per sigma level."""
    return FIXED_THRESH_MAP.get(sigma, BASE_CONF)

def threshold_auto(sigma: int) -> float:
    """
    Auto-adapt: linearly scale threshold down as sigma increases.
    At sigma=0   → 0.25 (baseline)
    At sigma=150 → 0.08 (most lenient)
    """
    max_sigma = 150
    min_thresh = 0.08
    max_thresh = 0.25
    ratio = sigma / max_sigma if max_sigma > 0 else 0
    return round(max_thresh - ratio * (max_thresh - min_thresh), 4)

# ── Helper: add Gaussian noise ─────────────────────────────────────────────────
def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return image.copy()
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# ── Helper: run model ──────────────────────────────────────────────────────────
def run_model(model, image_bgr: np.ndarray, conf_thresh: float):
    results = model(image_bgr, verbose=False, conf=conf_thresh)[0]
    boxes   = results.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0, 0.0
    confs = boxes.conf.cpu().numpy()
    return int(len(confs)), float(confs.mean()), float(confs.max())

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading model : {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
all_images = sorted([
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
])

if not all_images:
    raise FileNotFoundError(f"No images found in: {IMG_DIR}")

total_images = len(all_images)
print(f"Total images  : {total_images}")
print(f"Sigma levels  : {SIGMAS}\n")

# ── Define all pipeline combinations ──────────────────────────────────────────
PREPROCESSORS = {
    "None"     : preprocess_none,
    "Gaussian" : preprocess_gaussian,
    "Median"   : preprocess_median,
    "NLM"      : preprocess_nlm,
}

THRESH_STRATEGIES = {
    "Fixed"    : threshold_fixed,
    "Auto"     : threshold_auto,
}

# ── Run all combinations ───────────────────────────────────────────────────────
all_rows = []

for prep_name, prep_fn in PREPROCESSORS.items():
    for thresh_name, thresh_fn in THRESH_STRATEGIES.items():

        combo_name = f"{prep_name} + {thresh_name}"
        print(f"\n{'─'*60}")
        print(f"  Pipeline: {combo_name}")
        print(f"{'─'*60}")

        baseline_conf = None
        combo_rows    = []

        for sigma in SIGMAS:
            conf_thresh      = thresh_fn(sigma)
            img_detected     = 0
            img_not_detected = 0
            mean_confs       = []
            max_confs        = []

            for img_path in all_images:
                orig = cv2.imread(img_path)
                if orig is None:
                    continue

                noisy     = add_gaussian_noise(orig, sigma)
                processed = prep_fn(noisy)

                n_det, mean_c, max_c = run_model(model, processed, conf_thresh)

                mean_confs.append(mean_c)
                max_confs.append(max_c)

                if n_det > 0:
                    img_detected += 1
                else:
                    img_not_detected += 1

            avg_mean_conf = float(np.mean(mean_confs)) if mean_confs else 0.0
            avg_max_conf  = float(np.mean(max_confs))  if max_confs  else 0.0
            det_rate      = (img_detected / total_images) * 100

            if sigma == 0:
                baseline_conf = avg_mean_conf

            degradation = 0.0
            if baseline_conf and baseline_conf > 0:
                degradation = ((baseline_conf - avg_mean_conf) / baseline_conf) * 100

            print(f"  σ={sigma:>4} | thresh={conf_thresh:.2f} | "
                  f"detected={img_detected}/{total_images} | "
                  f"mean_conf={avg_mean_conf:.4f} | "
                  f"det_rate={det_rate:.2f}% | "
                  f"degradation={degradation:.2f}%")

            combo_rows.append({
                "Pipeline"            : combo_name,
                "Preprocessor"        : prep_name,
                "Threshold Strategy"  : thresh_name,
                "Sigma (σ)"           : sigma,
                "Conf Threshold Used" : conf_thresh,
                "Total Images"        : total_images,
                "Images Detected"     : img_detected,
                "Images Not Detected" : img_not_detected,
                "Mean Confidence"     : round(avg_mean_conf, 4),
                "Max Confidence"      : round(avg_max_conf,  4),
                "Detection Rate (%)"  : round(det_rate,      2),
                "Degradation (%)"     : round(degradation,   2),
            })

        all_rows.extend(combo_rows)

# ── Save table ────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

csv_path  = os.path.join(OUT_DIR, "improvement_results.csv")
xlsx_path = os.path.join(OUT_DIR, "improvement_results.xlsx")

df.to_csv(csv_path, index=False)
print(f"\nCSV saved  → {csv_path}")

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="All Results")
    ws = writer.sheets["All Results"]
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4

    # Also write one sheet per preprocessor
    for prep_name in PREPROCESSORS:
        sub = df[df["Preprocessor"] == prep_name]
        sub.to_excel(writer, index=False, sheet_name=prep_name)

print(f"Excel saved → {xlsx_path}")

# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = {
    "None + Fixed"     : "#95A5A6",
    "None + Auto"      : "#7F8C8D",
    "Gaussian + Fixed" : "#2980B9",
    "Gaussian + Auto"  : "#85C1E9",
    "Median + Fixed"   : "#27AE60",
    "Median + Auto"    : "#82E0AA",
    "NLM + Fixed"      : "#8E44AD",
    "NLM + Auto"       : "#C39BD3",
}
STYLES = {
    "Fixed" : "-",
    "Auto"  : "--",
}

pipelines = df["Pipeline"].unique()

# ── Plot 1: Detection Rate % vs Sigma ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 6))
fig1.suptitle("Detection Rate % vs Gaussian Noise — All Pipelines",
              fontsize=13, fontweight="bold")

for pipe in pipelines:
    sub = df[df["Pipeline"] == pipe].sort_values("Sigma (σ)")
    thresh = pipe.split(" + ")[1]
    ax1.plot(sub["Sigma (σ)"], sub["Detection Rate (%)"],
             marker="o", lw=2, linestyle=STYLES[thresh],
             color=COLORS[pipe], label=pipe)

ax1.set_xlabel("Gaussian Noise σ", fontsize=11)
ax1.set_ylabel("Detection Rate (%)", fontsize=11)
ax1.set_ylim(0, 110)
ax1.set_yticks(range(0, 110, 10))
ax1.set_xticks(SIGMAS)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(fontsize=8, loc="upper right", ncol=2)
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "plot1_detection_rate_comparison.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
print(f"Plot 1 saved → {p1}")

# ── Plot 2: Mean Confidence vs Sigma ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.suptitle("Mean Confidence vs Gaussian Noise — All Pipelines",
              fontsize=13, fontweight="bold")

for pipe in pipelines:
    sub = df[df["Pipeline"] == pipe].sort_values("Sigma (σ)")
    thresh = pipe.split(" + ")[1]
    ax2.plot(sub["Sigma (σ)"], sub["Mean Confidence"],
             marker="o", lw=2, linestyle=STYLES[thresh],
             color=COLORS[pipe], label=pipe)

ax2.set_xlabel("Gaussian Noise σ", fontsize=11)
ax2.set_ylabel("Mean Confidence", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.set_xticks(SIGMAS)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(fontsize=8, loc="upper right", ncol=2)
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "plot2_confidence_comparison.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"Plot 2 saved → {p2}")

# ── Plot 3: Degradation % vs Sigma ────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(13, 6))
fig3.suptitle("Degradation % vs Gaussian Noise — All Pipelines",
              fontsize=13, fontweight="bold")

for pipe in pipelines:
    sub = df[df["Pipeline"] == pipe].sort_values("Sigma (σ)")
    thresh = pipe.split(" + ")[1]
    ax3.plot(sub["Sigma (σ)"], sub["Degradation (%)"],
             marker="o", lw=2, linestyle=STYLES[thresh],
             color=COLORS[pipe], label=pipe)

ax3.axhline(50, color="#E74C3C", linestyle=":", lw=1.8, label="50% degradation mark")
ax3.set_xlabel("Gaussian Noise σ", fontsize=11)
ax3.set_ylabel("Degradation (%)", fontsize=11)
ax3.set_ylim(0, 110)
ax3.set_yticks(range(0, 110, 10))
ax3.set_xticks(SIGMAS)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(fontsize=8, loc="upper left", ncol=2)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "plot3_degradation_comparison.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"Plot 3 saved → {p3}")

# ── Plot 4: Best pipeline per sigma (Detection Rate bar chart) ────────────────
fig4, ax4 = plt.subplots(figsize=(13, 6))
fig4.suptitle("Best Detection Rate per Sigma — Pipeline Comparison",
              fontsize=13, fontweight="bold")

x      = np.arange(len(SIGMAS))
width  = 0.10
n_pipes = len(pipelines)
offsets = np.linspace(-(n_pipes-1)/2, (n_pipes-1)/2, n_pipes) * width

for i, pipe in enumerate(pipelines):
    sub  = df[df["Pipeline"] == pipe].sort_values("Sigma (σ)")
    vals = sub["Detection Rate (%)"].tolist()
    ax4.bar(x + offsets[i], vals, width=width,
            color=COLORS[pipe], alpha=0.85, label=pipe)

ax4.set_xlabel("Gaussian Noise σ", fontsize=11)
ax4.set_ylabel("Detection Rate (%)", fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels([f"σ={s}" for s in SIGMAS])
ax4.set_ylim(0, 115)
ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
ax4.legend(fontsize=8, loc="upper right", ncol=2)
plt.tight_layout()
p4 = os.path.join(OUT_DIR, "plot4_best_pipeline_per_sigma.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
print(f"Plot 4 saved → {p4}")

plt.show()

# ── Print best pipeline per sigma ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Best Pipeline per Sigma Level (by Detection Rate %)")
print(f"{'='*60}")
for sigma in SIGMAS:
    sub  = df[df["Sigma (σ)"] == sigma]
    best = sub.loc[sub["Detection Rate (%)"].idxmax()]
    print(f"  σ={sigma:>4} → {best['Pipeline']:<22} "
          f"det_rate={best['Detection Rate (%)']:.2f}%  "
          f"mean_conf={best['Mean Confidence']:.4f}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")