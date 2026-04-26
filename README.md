# Diagnostic Analysis of YOLO Architecture: Kirana-Kosh

This repository contains the reproducible experiments for Phase 2 of the Deep Learning for Computer Vision (DLCV) course. The project probes a YOLOv8-Nano model trained on the custom `Kirana-Kosh` dataset to understand its inductive biases, scale dependencies, and few-shot failure modes in highly dense Indian retail environments.

## Project Members
* Pranav Khunt (23ucs671)
* Jeet Manseta (23ucs603)
* Advait Karia (23ucs520)

## Repository Contents
* `Experiments.ipynb`: A unified Google Colab notebook containing the code for all three experimental units.
* `requirements.txt`: Python dependencies required to run the experiments.
* `report.pdf`: The final 4-6 page diagnostic analysis report.

## Experimental Units Covered
1. **Unit I (Data Augmentation as Inductive Bias):** Evaluates model robustness by applying heavy Gaussian blur to the test set to determine if augmentation teaches low-frequency structural invariances.
2. **Unit II (Feature Pyramid Surgery):** Mathematically ablates specific detection heads (P3, P4, P5) in the YOLOv8 Feature Pyramid Network to map scale-specific architectural dependencies.
3. **Unit III (Few-Shot Learning Failure Modes):** Plots per-class mAP50 against training instance counts to establish a threshold where detection capacity collapses due to intra-class variance.

## How to Run
The experiments are designed to be run sequentially in Google Colab (T4 GPU recommended). 

1. Clone this repository.
2. Open `Experiments.ipynb` in Google Colab.
3. Install the required dependencies: `pip install -r requirements.txt`.
4. Run the notebook cells sequentially. *(Note: You will need your own Roboflow API key to fetch the dataset).*