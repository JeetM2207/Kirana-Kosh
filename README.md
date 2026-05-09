# Kirana-Kosh: Diagnostic Analysis of YOLOv8 on Indian Retail Environments

> *Treating the model as a scientific subject — not a black box.*

Standard object detection benchmarks were built on Western datasets. They fail silently in
Indian Kirana stores — where shelves are packed wall-to-wall, a matchbox sits next to a 5kg
rice bag, and products overlap in every direction. This project doesn't just train a model and
report mAP. It **probes** a YOLOv8-Nano architecture to understand *why* it works, *where*
it breaks, and *what* drives each failure.

---

## Project Members

| Name | Roll No |
|------|---------|
| Pranav Khunt | 23ucs671 |
| Jeet Manseta | 23ucs603 |
| Advait Karia | 23ucs520 |

*Deep Learning for Computer Vision (DLCV) — April 2026*

---

## The Dataset: Kirana-Kosh

We built a custom dataset from scratch because no existing benchmark captures the visual
complexity of real Indian retail environments.

- **1,189 curated images** sourced from Indian Kirana stores
- **22 Super-Classes** designed to maximize inter-class visual separation
- Covers: Beverages, Biscuits & Cookies, Chips & Wafers, Chocolates, Cooking Oil, Dairy &
  Ghee, Spices & Masala, Tea & Coffee, Namkeen & Snacks, Health Drinks, Hair Care, Oral
  Care, Household Cleaning, Instant Food, Dry Fruits & Nuts, Grains & Flour, Bath & Body,
  Deodorants, Baby Care, Pharma, Sauces & Spreads, Other Packaged

**Why this dataset is hard:**
- Extreme shelf density — dozens of items per image
- Massive scale variance — tiny matchboxes alongside 5kg rice bags
- Heavy item overlap and occlusion
- Inconsistent lighting across store environments

---

## Experimental Units

### Unit I — Data Augmentation as Inductive Bias

**Question:** What invariances does augmentation actually teach the model?

**Setup:** Two identical YOLOv8-Nano models were trained — a clean Baseline and an
Augmented model (3x dataset with ±15° rotation, brightness variations, and blur). Both
were then evaluated on a *deliberately corrupted* test set with heavy Gaussian blur applied
via OpenCV.

**Results:**

| Model | Global mAP50 | Health Drinks | Cooking Oil | Household Cleaning |
|-------|-------------|---------------|-------------|-------------------|
| Baseline | 14.5% | 28.0% | — | — |
| Augmented | **24.0%** | **56.8%** | — | — |

**Why it happens:** The Baseline overfit to high-frequency fragile features — sharp text,
crisp logo edges, barcode lines. Gaussian blur mathematically destroys these, causing
detection to collapse. The Augmented model was repeatedly trained on degraded images,
forcing it to learn low-frequency structural features like packet shape and aspect ratio —
features that survive real-world camera blur.

**Takeaway:** Augmentation is not just a performance booster. It is a deliberate inductive
bias mechanism that controls *which features* the model learns to rely on.

---

### Unit II — Feature Pyramid Surgery

**Question:** Which FPN detection head is architecturally critical for retail-scale products?

**Setup:** YOLOv8 uses a Feature Pyramid Network (FPN) with three detection heads — P3
for small objects, P4 for medium objects, and P5 for large objects. We performed surgical
ablation by mathematically zeroing out the `cv2` (bounding box) and `cv3` (classification)
branches of one head at a time, then running full inference to measure the impact.

**Results:**

| Ablated Head | Global mAP50 | Key Class Failure |
|-------------|-------------|-------------------|
| None (Baseline) | 30.0% | — |
| P3 (Small) | 18.4% | Other Packaged Goods: 2.3% → 0.6% |
| **P4 (Medium)** | **8.2%** | **Cooking Oil: 52.8% → 1.5%** |
| P5 (Large) | 14.4% | Namkeen & Snacks: 44.9% → 12.5% |

**Why it happens:** Retail photos are taken from a standard shopping distance. This means
the vast majority of products fall into the medium pixel-area range — exactly what P4
processes. Removing P4 makes the model functionally blind to standard shelf inventory,
skipping entire rows of products during inference.

**Takeaway:** P4 is the architectural backbone of this model for retail detection. Its
criticality is completely invisible to standard mAP evaluation — you only discover it by
looking inside the architecture.

---

### Unit III — Few-Shot Learning Failure Modes

**Question:** At what point does the model give up on a class entirely?

**Setup:** Per-class Average Precision was extracted from the Augmented model's baseline
evaluation and plotted against the exact number of training instances for each of the 22
classes.

**Results:**
- Classes with fewer than **75 training instances** consistently fall into a failure zone
- Health & Pharma (30 instances) achieved only ~3.0% mAP50
- Chocolates & Sweets (69 instances) also failed at only 2.3% mAP50 — *despite being
  close to the 75-instance threshold*

**Why Chocolates fail despite 69 instances:** Chocolates in India range from tiny candy
bars to large assorted gift boxes with completely different shapes, sizes, colors, and
wrappers. The intra-class visual variance is so high that 69 instances cannot cover the
feature distribution space. Meanwhile, visually uniform classes like Cooking Oil (mostly
identical rectangular bottles) succeed at similar instance counts.

**Takeaway:** Few-shot failure is not purely a function of instance count. It is governed
by intra-class visual variance. Data sufficiency must be measured per class by visual
diversity — not raw numbers alone.

---

## Key Findings Summary

| Unit | Finding |
|------|---------|
| I — Augmentation | Teaches structural invariance; forces abandonment of fragile high-frequency features |
| II — FPN Surgery | P4 is the core engine; its removal causes 30.0% → 8.2% mAP collapse |
| III — Few-Shot | Visual diversity within a class matters more than instance count alone |

---

## Real-World Deployment Potential

The system is designed for direct integration with existing Kirana store CCTV infrastructure
for automated, real-time inventory tracking and shelf compliance monitoring — no new hardware
required. Standard pretrained models fail in this environment. The diagnostic-first approach
taken here ensures a robust, India-specific solution built on a genuine understanding of the
model's failure modes.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Experiments.ipynb` | Unified Google Colab notebook with all three experimental units |
| `requirements.txt` | Python dependencies |
| `report.pdf` | Full 8-page diagnostic analysis report |

---

## How to Run

The experiments are designed for Google Colab with a T4 GPU.

```bash
# 1. Clone the repository
git clone <your-repo-url>

# 2. Open Experiments.ipynb in Google Colab

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run cells sequentially
```

> **Note:** You will need your own Roboflow API key to fetch the Kirana-Kosh dataset.
> Set it when prompted in the notebook.

---

## Tech Stack

- **Model:** YOLOv8-Nano (Ultralytics)
- **Dataset Pipeline:** Roboflow
- **Image Corruption:** OpenCV (Gaussian Blur)
- **Augmentation:** Ultralytics built-in + manual 3x multiplication
- **Analysis & Visualization:** Python, Matplotlib, NumPy, Pandas
- **Training Environment:** Google Colab (T4 GPU)
