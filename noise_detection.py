"""
YOLO26n — Gaussian Noise Threshold Test (Rewritten))
=====================================================
- Loads all images from the dataset directory
- Applies Gaussian noise from sigma 0 to 150
- Runs yolo26n.onnx on each noisy image
- Records detection counts and confidence scores
- Finds breaking point at 50% confidence degradation
- Saves annotated images only for the first 100 images (visual subset)
- Generates two separate plots:
    1. Confidence score degradation
    2. Average detection rate

Run from inside your cloned repo:
    python noise_threshold_26n.py

Output:
    noise_threshold_results/
        sigma_0/  sigma_10/ ... sigma_150/   ← annotated (first 100 images only)
    confidence_plot.png
    detection_rate_plot.png
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = #path to image dataset
OUT_DIR    = os.path.join(BASE, "noise_threshold_results")
MODEL_PATH = os.path.join(BASE, "yolo26n.onnx")

CONF_THRESH          = 0.25                             # Minimum confidence to count a detection
SIGMAS               = list(range(0, 151, 25))          # [0, 10, 20, ..., 150]
DEGRADATION_THRESH   = 0.50                             # Break when conf drops to 50% of baseline
MAX_SAVE_IMAGES      = 100                              # Save annotated output for first N images only

# ── Helper: add Gaussian noise ─────────────────────────────────────────────────
def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise with std-dev `sigma` to a BGR image."""
    if sigma == 0:
        return image.copy()
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ── Helper: run model on one image ─────────────────────────────────────────────
def run_model(model, image_bgr: np.ndarray):
    """
    Returns:
        n_det   (int)   — number of detections above CONF_THRESH
        mean_c  (float) — mean confidence across detections (0.0 if none)
        max_c   (float) — max  confidence across detections (0.0 if none)
    """
    results = model(image_bgr, verbose=False, conf=CONF_THRESH)[0]
    boxes   = results.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0, 0.0
    confs = boxes.conf.cpu().numpy()
    return int(len(confs)), float(confs.mean()), float(confs.max())


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading model : {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")

# ── Collect all images ─────────────────────────────────────────────────────────
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
all_images = sorted([
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
])

if not all_images:
    raise FileNotFoundError(f"No images found in: {IMG_DIR}")

print(f"Total images  : {len(all_images)}")
print(f"Sigma levels  : {SIGMAS}")
print(f"Images saved  : first {MAX_SAVE_IMAGES} per sigma level\n")

# ── Main evaluation loop ───────────────────────────────────────────────────────
results_per_sigma = []   # one dict per sigma level

for sigma in SIGMAS:
    sigma_dir = os.path.join(OUT_DIR, f"sigma_{sigma}")
    os.makedirs(sigma_dir, exist_ok=True)

    det_counts = []
    mean_confs = []
    max_confs  = []

    for idx, img_path in enumerate(all_images):
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"  [WARN] Could not read: {img_path}")
            continue

        noisy              = add_gaussian_noise(orig, sigma)
        n_det, mean_c, max_c = run_model(model, noisy)

        det_counts.append(n_det)
        mean_confs.append(mean_c)
        max_confs.append(max_c)

        # ── Save annotated image for the first MAX_SAVE_IMAGES only ──────────
        if idx < MAX_SAVE_IMAGES:
            vis_result = model(noisy, verbose=False, conf=CONF_THRESH)[0]
            annotated  = vis_result.plot()
            label      = f"sigma={sigma}  dets={n_det}  conf={mean_c:.2f}"
            cv2.putText(annotated, label,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            save_path  = os.path.join(sigma_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, annotated)

    # Aggregate across all images for this sigma
    avg_dets = float(np.mean(det_counts)) if det_counts else 0.0
    avg_conf = float(np.mean(mean_confs)) if mean_confs else 0.0
    avg_max  = float(np.mean(max_confs))  if max_confs  else 0.0

    results_per_sigma.append({
        "sigma"          : sigma,
        "avg_detections" : round(avg_dets, 3),
        "avg_mean_conf"  : round(avg_conf, 4),
        "avg_max_conf"   : round(avg_max,  4),
    })

    print(f"  σ={sigma:>4} | dets={avg_dets:>6.3f} | "
          f"mean_conf={avg_conf:.4f} | max_conf={avg_max:.4f}")

# ── Find the breaking-point sigma ─────────────────────────────────────────────
baseline_conf   = results_per_sigma[0]["avg_mean_conf"]
threshold_sigma = None

for r in results_per_sigma[1:]:
    if baseline_conf > 0 and r["avg_mean_conf"] <= baseline_conf * DEGRADATION_THRESH:
        threshold_sigma = r["sigma"]
        break

# Fallback: first sigma where no detections at all
if threshold_sigma is None:
    for r in results_per_sigma:
        if r["avg_detections"] == 0:
            threshold_sigma = r["sigma"]
            break

print(f"\n{'='*55}")
print(f"  Baseline confidence (σ=0) : {baseline_conf:.4f}")
if threshold_sigma is not None:
    print(f"  Breaking point            : σ = {threshold_sigma}  "
          f"(≥50% confidence drop)")
else:
    print("  Breaking point NOT reached within σ=0–150.")
print(f"{'='*55}\n")

# ── Average degradation rate ───────────────────────────────────────────────
print("\nDegradation Rate (confidence drop per unit σ):")
for i in range(1, len(results_per_sigma)):
    prev = results_per_sigma[i - 1]
    curr = results_per_sigma[i]
    delta_conf  = prev["avg_mean_conf"] - curr["avg_mean_conf"]
    delta_sigma = curr["sigma"] - prev["sigma"]
    rate = delta_conf / delta_sigma if delta_sigma > 0 else 0
    print(f"  σ {prev['sigma']:>3} → {curr['sigma']:>3} : {rate:.6f} conf/σ")

total_drop = results_per_sigma[0]["avg_mean_conf"] - results_per_sigma[-1]["avg_mean_conf"]
total_sigma_range = results_per_sigma[-1]["sigma"] - results_per_sigma[0]["sigma"]
avg_rate = total_drop / total_sigma_range if total_sigma_range > 0 else 0
print(f"\n  Overall avg degradation rate: {avg_rate:.6f} conf/σ")
print(f"  Total confidence drop (σ=0 → σ={results_per_sigma[-1]['sigma']}): {total_drop:.4f}")

# ── Prepare series for plotting ────────────────────────────────────────────────
sigmas    = [r["sigma"]           for r in results_per_sigma]
avg_confs = [r["avg_mean_conf"]   for r in results_per_sigma]
avg_maxcs = [r["avg_max_conf"]    for r in results_per_sigma]
avg_dets  = [r["avg_detections"]  for r in results_per_sigma]

THRESHOLD_LINE = baseline_conf * DEGRADATION_THRESH   # 50% of baseline

# ── Plot 1: Confidence Score Degradation ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 5))
fig1.suptitle(
    "YOLO26n — Confidence Score vs Gaussian Noise\n(Indian Road Sign Detection)",
    fontsize=13, fontweight="bold"
)

ax1.plot(sigmas, avg_confs, "o-",  color="#2980B9", lw=2.5, ms=7,
         label="Mean confidence")
ax1.plot(sigmas, avg_maxcs, "s--", color="#85C1E9", lw=1.8, ms=5,
         label="Max confidence")
ax1.fill_between(sigmas, avg_confs, alpha=0.12, color="#2980B9")

# 50%-of-baseline horizontal reference
ax1.axhline(THRESHOLD_LINE, color="#E74C3C", linestyle=":", lw=1.8,
            label=f"50% of baseline ({THRESHOLD_LINE:.3f})")

# Vertical breaking-point line
if threshold_sigma is not None:
    ax1.axvline(threshold_sigma, color="#E74C3C", linestyle="--", lw=2,
                label=f"Breaking point σ={threshold_sigma}")
    ax1.annotate(
        f"Breaking point\nσ={threshold_sigma}",
        xy=(threshold_sigma, THRESHOLD_LINE),
        xytext=(threshold_sigma + 6, THRESHOLD_LINE + 0.08),
        fontsize=9, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.4)
    )

ax1.set_xlabel("Gaussian Noise σ (std dev, 0–150 scale)", fontsize=11)
ax1.set_ylabel("Confidence Score", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_xticks(sigmas)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(fontsize=9, loc="upper right")

plt.tight_layout()
conf_plot_path = os.path.join(BASE, "confidence_plot.png")
fig1.savefig(conf_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Confidence plot saved → {conf_plot_path}")

# ── Plot 2: Average Detection Rate ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.suptitle(
    "YOLO26n — Average Detection Rate vs Gaussian Noise\n(Indian Road Sign Detection)",
    fontsize=13, fontweight="bold"
)

ax2.bar(sigmas, avg_dets, width=7, color="#27AE60", alpha=0.70,
        label="Avg detections / image", zorder=2)
ax2.plot(sigmas, avg_dets, "o-", color="#1A5C38", lw=2.2, ms=6,
         label="Trend", zorder=3)

if threshold_sigma is not None:
    ax2.axvline(threshold_sigma, color="#E74C3C", linestyle="--", lw=2,
                label=f"Breaking point σ={threshold_sigma}")
    ax2.annotate(
        f"Breaking point\nσ={threshold_sigma}",
        xy=(threshold_sigma, max(avg_dets) * 0.5),
        xytext=(threshold_sigma + 5, max(avg_dets) * 0.6),
        fontsize=9, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.4)
    )

ax2.set_xlabel("Gaussian Noise σ (std dev, 0–150 scale)", fontsize=11)
ax2.set_ylabel("Avg Detections per Image", fontsize=11)
ax2.set_xticks(sigmas)
ax2.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
ax2.legend(fontsize=9, loc="upper right")

plt.tight_layout()
det_plot_path = os.path.join(BASE, "detection_rate_plot.png")
fig2.savefig(det_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Detection rate plot saved → {det_plot_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\nAnnotated images (first {MAX_SAVE_IMAGES}) saved → {OUT_DIR}/")
print("Done.")
