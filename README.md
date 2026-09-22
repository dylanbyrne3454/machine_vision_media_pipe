# MediaPipe Kinematic Tracking & 3D World Coordinate Benchmarking

An end-to-end Python processing and evaluation pipeline for monocular markerless motion capture (MMC). This repository extends a benchmark framework by evaluating **MediaPipe Pose** (2D normalised vs. 3D real-world coordinates in meters) against 100 Hz optical motion capture (OMC) ground truth.

---

## Context & Motivation

Extends prior OpenPose work by upgrading the pipeline to MediaPipe Pose to benchmark 2D and native 3D world coordinates against optical motion capture ground truth.


## Repository Structure

```text
├── data/
│   ├── keypoints/                # Normalised and world coordinates
│   ├── pickle/                   # OMC ground truths
│   └── dj_manual_segs.json       # Manual segmentation metadata
├── notebooks/
│   ├── archive/                  # Legacy/experimental notebooks
│   ├── 01_mediapipe_normalised.ipynb
│   ├── 02_full_coordinate_comparison_screening.ipynb
│   ├── 03_full_cooridnate_comparison.ipynb
│   └── posedata.py               # Pose data handling utilities
├── results/                      # Output plots and benchmarking metrics
├── tasks/                        # Motor task scripts
│   ├── cmj.py                    # Countermovement Jump
│   ├── dj.py                     # Drop Jump
│   ├── hip.py                    # Hip mobility
│   ├── nordic.py                 # Nordic Hamstring exercise
│   ├── rjt.py                    # Repeated Jump Test
│   ├── slr.py                    # Straight Leg Raise
│   ├── sls.py                    # Single Leg Squat
│   └── velocity.py               # Barbell/movement velocity tracking
├── utilities/                    # Core helper routines
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies
```

---

## Methodology & Signal Processing

1. **Multi-Dimensional Keypoint Extraction:**
   Extracts 33 body landmarks frame-by-frame using MediaPipe Pose (`model_complexity=2`), yielding:
   * **2D Normalised Coordinates:** Keypoint coordinates scaled [0.0, 1.0] relative to image width/height.
   * **3D World Coordinates:** Real-world 3D coordinates in meters centered at the subject's hip midpoint.

2. **Signal Smoothing:**
   Extracted keypoint series are smoothed using a second-order **Savitzky-Golay filter** to preserve movement extrema (e.g., peak jump height, maximum range of motion) while filtering high-frequency jitter.

3. **FFT Resampling & Synchronization:**
   To align 30 fps smartphone video streams with 100 Hz Codamotion OMC ground truth, time-series data are upsampled using **Fast Fourier Transform (FFT) resampling** to minimize signal distortion.

4. **Repetition Segmentation:**
   Automatically segments repetition windows.
---

## Key Results

Range of motion (ROM) metrics were evaluated across tasks against 100 Hz optical motion capture (OMC) ground truth. Performance was quantified using Mean Absolute Error (MAE in degrees) across four MediaPipe coordinate representations: **2D Normalised**, **2D World**, **3D Normalised**, and **3D World**.

![Range of Motion MAE Matrix](results/coord_results_final.png)

### MAE Breakdown (vs. OMC Ground Truth)

| Movement Task | 2D-norm (° MAE) | 2D-world (° MAE) | 3D-norm (° MAE) | 3D-world (° MAE) |
| :--- | :---: | :---: | :---: | :---: |
| **Hip external rotation** | **8.4** | 8.8 | 25.4 | 12.2 |
| **Hip internal rotation** | 21.4 | 14.5 | 28.4 | **12.2** |
| **Nordic curl** | **11.9** | 17.1 | 22.7 | 16.8 |
| **Single leg squat** | 22.4 | 22.9 | **21.9** | 25.1 |
| **Straight leg raise** | 10.3 | **5.1** | 10.2 | **5.4** |

---

## Quick Start

### Installation
```bash
git clone [https://github.com/dylanbyrne3454/machine_vision_media_pipe.git](https://github.com/dylanbyrne3454/machine_vision_media_pipe.git)
cd machine_vision_media_pipe
pip install -r requirements.txt
```

### Dependencies
* `opencv-python`
* `mediapipe`
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

---

## 📖 Reference & Prior Work

This repository extends the baseline pipeline introduced in:
* **Aderinola et al. (2023)** — *Machine Vision-Enabled Sports Performance Analysis* ([arXiv:2312.11340](https://arxiv.org/abs/2312.11340))
