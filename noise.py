"""
YOLO26n — Gaussian Noise Analysis (Full Pipeline)
==================================================
- Loads all images from dataset directory
- Applies Gaussian noise at sigma: 0, 25, 50, 75, 100, 125, 150
- Runs yolo26n.onnx on each noisy image
- Records per-sigma stats:
    total images, detected, not detected,
    mean conf, max conf, degradation %, detection rate %
- Saves results as CSV + Excel
- Generates 4 plots:
    1. Confidence vs Gaussian Noise
    2. Detection Rate % vs Gaussian Noise
    3. Degradation % vs Gaussian Noise
    4. Degradation Rate between sigma steps (bar chart)

Run:
    python noise_analysis.py
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
OUT_DIR    = os.path.join(BASE, "noise_analysis_results")
MODEL_PATH = os.path.join(BASE, "yolo26n.onnx")

CONF_THRESH        = 0.25
SIGMAS             = [0, 25, 50, 75, 100, 125, 150]
DEGRADATION_THRESH = 0.50    # 50% of baseline = breaking point

# ── Helper: add Gaussian noise ─────────────────────────────────────────────────
def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return image.copy()
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# ── Helper: run model on one image ─────────────────────────────────────────────
def run_model(model, image_bgr: np.ndarray):
    results = model(image_bgr, verbose=False, conf=CONF_THRESH)[0]
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

# ── Main evaluation loop ───────────────────────────────────────────────────────
rows = []

for sigma in SIGMAS:
    det_counts  = []
    mean_confs  = []
    max_confs   = []
    img_detected     = 0
    img_not_detected = 0

    for img_path in all_images:
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"  [WARN] Could not read: {img_path}")
            continue

        noisy = add_gaussian_noise(orig, sigma)
        n_det, mean_c, max_c = run_model(model, noisy)

        det_counts.append(n_det)
        mean_confs.append(mean_c)
        max_confs.append(max_c)

        if n_det > 0:
            img_detected += 1
        else:
            img_not_detected += 1

    avg_mean_conf = float(np.mean(mean_confs)) if mean_confs else 0.0
    avg_max_conf  = float(np.mean(max_confs))  if max_confs  else 0.0
    avg_det_rate  = (img_detected / total_images) * 100

    rows.append({
        "Sigma (σ)"              : sigma,
        "Total Images"           : total_images,
        "Images Detected"        : img_detected,
        "Images Not Detected"    : img_not_detected,
        "Mean Confidence"        : round(avg_mean_conf, 4),
        "Max Confidence"         : round(avg_max_conf,  4),
        "Detection Rate (%)"     : round(avg_det_rate,  2),
        "Degradation (%)"        : None,   # filled below
    })

    print(f"  σ={sigma:>4} | detected={img_detected}/{total_images} | "
          f"mean_conf={avg_mean_conf:.4f} | max_conf={avg_max_conf:.4f} | "
          f"det_rate={avg_det_rate:.2f}%")

# ── Compute degradation % relative to baseline ────────────────────────────────
baseline_conf = rows[0]["Mean Confidence"]

for r in rows:
    if baseline_conf > 0:
        drop = ((baseline_conf - r["Mean Confidence"]) / baseline_conf) * 100
        r["Degradation (%)"] = round(drop, 2)
    else:
        r["Degradation (%)"] = 0.0

# ── Find breaking point ───────────────────────────────────────────────────────
threshold_sigma = None
for r in rows[1:]:
    if r["Mean Confidence"] <= baseline_conf * DEGRADATION_THRESH:
        threshold_sigma = r["Sigma (σ)"]
        break

print(f"\nBaseline confidence (σ=0) : {baseline_conf:.4f}")
if threshold_sigma:
    print(f"Breaking point            : σ = {threshold_sigma}  (≥50% confidence drop)")
else:
    print("Breaking point NOT reached within tested sigma range.")

# ── Save table ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

csv_path   = os.path.join(OUT_DIR, "noise_analysis_results.csv")
xlsx_path  = os.path.join(OUT_DIR, "noise_analysis_results.xlsx")

df.to_csv(csv_path, index=False)
print(f"\nCSV saved  → {csv_path}")

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Noise Analysis")
    ws = writer.sheets["Noise Analysis"]

    # Auto-fit column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4

print(f"Excel saved → {xlsx_path}")

# ── Prepare series ────────────────────────────────────────────────────────────
sigmas      = df["Sigma (σ)"].tolist()
mean_confs  = df["Mean Confidence"].tolist()
max_confs   = df["Max Confidence"].tolist()
det_rates   = df["Detection Rate (%)"].tolist()
degradation = df["Degradation (%)"].tolist()

threshold_line = baseline_conf * DEGRADATION_THRESH

# Degradation RATE between steps (delta per unit sigma)
deg_rates       = []
deg_rate_labels = []
for i in range(1, len(rows)):
    prev = rows[i - 1]
    curr = rows[i]
    delta_conf  = prev["Mean Confidence"] - curr["Mean Confidence"]
    delta_sigma = curr["Sigma (σ)"] - prev["Sigma (σ)"]
    rate = (delta_conf / delta_sigma) * 100 if delta_sigma > 0 else 0
    deg_rates.append(round(rate, 4))
    deg_rate_labels.append(f"σ{prev['Sigma (σ)']}→{curr['Sigma (σ)']}")

# ── Plot helpers ──────────────────────────────────────────────────────────────
BLUE   = "#2980B9"
LBLUE  = "#85C1E9"
GREEN  = "#27AE60"
RED    = "#E74C3C"
ORANGE = "#F39C12"
PURPLE = "#8E44AD"

def add_threshold_vline(ax, ts):
    if ts is not None:
        ax.axvline(ts, color=RED, linestyle="--", lw=1.8,
                   label=f"Breaking point σ={ts}")

# ── Plot 1: Confidence vs Gaussian Noise ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 5))
fig1.suptitle("YOLO26n — Confidence vs Gaussian Noise\n(Indian Road Sign Detection)",
              fontsize=13, fontweight="bold")

ax1.plot(sigmas, mean_confs, "o-",  color=BLUE,  lw=2.5, ms=7, label="Mean confidence")
ax1.plot(sigmas, max_confs,  "s--", color=LBLUE, lw=1.8, ms=5, label="Max confidence")
ax1.fill_between(sigmas, mean_confs, alpha=0.12, color=BLUE)

ax1.axhline(threshold_line, color=RED,    linestyle=":",  lw=1.8,
            label=f"50% of baseline ({threshold_line:.3f})")
ax1.axhline(CONF_THRESH,    color=ORANGE, linestyle="-.", lw=1.8,
            label=f"Detection threshold ({CONF_THRESH})")

add_threshold_vline(ax1, threshold_sigma)

ax1.set_xlabel("Gaussian Noise σ", fontsize=11)
ax1.set_ylabel("Confidence Score", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_xticks(sigmas)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(fontsize=9, loc="upper right")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "plot1_confidence_vs_noise.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
print(f"Plot 1 saved → {p1}")

# ── Plot 2: Detection Rate % vs Gaussian Noise ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.suptitle("YOLO26n — Detection Rate % vs Gaussian Noise\n(Indian Road Sign Detection)",
              fontsize=13, fontweight="bold")

ax2.bar(sigmas, det_rates, width=12, color=GREEN, alpha=0.70,
        label="Detection rate %", zorder=2)
ax2.plot(sigmas, det_rates, "o-", color="#1A5C38", lw=2.2, ms=6,
         label="Trend", zorder=3)

add_threshold_vline(ax2, threshold_sigma)

ax2.set_xlabel("Gaussian Noise σ", fontsize=11)
ax2.set_ylabel("Detection Rate (%)", fontsize=11)
ax2.set_ylim(0, 110)
ax2.set_yticks(range(0, 110, 10))
ax2.set_xticks(sigmas)
ax2.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
ax2.legend(fontsize=9, loc="upper right")
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "plot2_detection_rate_vs_noise.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"Plot 2 saved → {p2}")

# ── Plot 3: Degradation % vs Gaussian Noise ───────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 5))
fig3.suptitle("YOLO26n — Cumulative Degradation % vs Gaussian Noise\n(Indian Road Sign Detection)",
              fontsize=13, fontweight="bold")

ax3.plot(sigmas, degradation, "o-", color=PURPLE, lw=2.5, ms=7,
         label="Degradation %")
ax3.fill_between(sigmas, degradation, alpha=0.12, color=PURPLE)

ax3.axhline(50, color=RED, linestyle=":", lw=1.8, label="50% degradation mark")
add_threshold_vline(ax3, threshold_sigma)

ax3.set_xlabel("Gaussian Noise σ", fontsize=11)
ax3.set_ylabel("Confidence Degradation (%)", fontsize=11)
ax3.set_ylim(0, 110)
ax3.set_yticks(range(0, 110, 10))
ax3.set_xticks(sigmas)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(fontsize=9, loc="upper left")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "plot3_degradation_vs_noise.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"Plot 3 saved → {p3}")

# ── Plot 4: Degradation Rate between sigma steps (bar chart) ──────────────────
fig4, ax4 = plt.subplots(figsize=(11, 5))
fig4.suptitle("YOLO26n — Degradation Rate Between Sigma Steps\n(Indian Road Sign Detection)",
              fontsize=13, fontweight="bold")

bar_colors = [RED if r == max(deg_rates) else ORANGE for r in deg_rates]
bars = ax4.bar(deg_rate_labels, deg_rates, color=bar_colors, alpha=0.80,
               edgecolor="white", lw=0.8)

# Value labels on top of bars
for bar, val in zip(bars, deg_rates):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#2C3E50")

ax4.set_xlabel("Sigma Step", fontsize=11)
ax4.set_ylabel("Confidence Drop per Unit σ (%)", fontsize=11)
ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
ax4.legend(handles=[
    plt.Rectangle((0,0),1,1, color=RED,    alpha=0.8, label="Highest drop step"),
    plt.Rectangle((0,0),1,1, color=ORANGE, alpha=0.8, label="Other steps"),
], fontsize=9, loc="upper right")
plt.tight_layout()
p4 = os.path.join(OUT_DIR, "plot4_degradation_rate_per_step.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
print(f"Plot 4 saved → {p4}")

plt.show()

# ── Print final table ─────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(df.to_string(index=False))
print(f"{'='*85}")
print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")