# MediaPipe Kinematic Tracking & 3D World Coordinate Benchmarking

An end-to-end Python processing and evaluation pipeline for monocular markerless motion capture (MMC). Developed at the **Insight SFI Centre for Data Analytics (UCD)**, this repository extends the benchmark framework established by Aderinola et al. (2023)[cite: 1] by evaluating **MediaPipe Pose** (2D normalized vs. 3D real-world coordinates in meters) against 100 Hz optical motion capture (OMC) ground truth[cite: 1].

---

## 📌 Context & Motivation

Prior work evaluated 2D OpenPose keypoints across 12 motor tasks using a single smartphone setup[cite: 1]. While 2D monocular tracking demonstrated high agreement for vertical jump height and barbell velocity, it showed inaccuracy when used for more advanced tasks including 3D angle metrics[cite: 1].

This repository addresses those limitations by:
1. Upgrading the processing pipeline to **MediaPipe Pose**.
2. Benchmarking both 2D normalised (x, y, z) and 3D camera-centric world coordinates (X, Y, Z in meters) against 3D optical motion capture ground truth.

---

## 🛠️ Repository Structure

```text
├── data/
│   ├── keypoints/                  # Normalised and world co-ordinates
│   ├── pickle/                     # OMC ground truths
│   └── dj_manual_segs.json         # Manual segmentation metadata
├── notebooks/
│   ├── archive/                    # Legacy/experimental notebooks
│   ├── 01_mediapipe_normalised.ipynb
│   ├── 02_full_coordinate_comparison_screening.ipynb
│   ├── 03_full_cooridnate_comparison.ipynb
│   └── posedata.py                 # Pose data handling utilities
├── results/                        # Output plots and benchmarking metrics
├── tasks/                          # Motor task scripts
│   ├── cmj.py                      # Countermovement Jump
│   ├── dj.py                       # Drop Jump
│   ├── hip.py                      # Hip mobility
│   ├── nordic.py                   # Nordic Hamstring exercise
│   ├── rjt.py                      # Repeated Jump Test
│   ├── slr.py                      # Straight Leg Raise
│   ├── sls.py                      # Single Leg Squat
│   └── velocity.py                 # Barbell/movement velocity tracking
├── utilities/                      # Core helper routines
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

---

## 📊 Methodology & Signal Processing

1. **Multi-Dimensional Keypoint Extraction:**
   Extracts 33 body landmarks frame-by-frame using MediaPipe Pose (`model_complexity=2`), yielding:
   * **2D Normalised Coordinates:** Keypoint coordinates scaled [0.0, 1.0] relative to image width/height.
   * **3D World Coordinates:** Real-world 3D coordinates in meters centered at the subject's hip midpoint.

2. **Signal Smoothing:**
   Extracted keypoint series are smoothed using a second-order **Savitzky-Golay filter**[cite: 1] to preserve movement extrema (e.g., peak jump height, maximum range of motion) while filtering high-frequency jitter[cite: 1].

3. **FFT Resampling & Synchronization:**
   To align 30 fps smartphone video streams with 100 Hz Codamotion OMC ground truth, time-series data are upsampled using **Fast Fourier Transform (FFT) resampling** to minimize signal distortion[cite: 1].

4. **Repetition Segmentation:**
   Automatically segments repetition windows around maximum displacement extrema[cite: 1].

---

## 🚀 Quick Start

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

This pipeline builds upon research conducted at the **Insight SFI Centre for Data Analytics, University College Dublin**:

> **Aderinola, T. B., Younesian, H., Goulding, C., Whelan, D., Caulfield, B., & Ifrim, G.** (2023). *Machine Vision-Enabled Sports Performance Analysis*. arXiv preprint arXiv:2312.11340[cite: 1].

---

## 👨‍💻 Author

**Dylan Byrne**  
*Undergraduate Electronic Engineering Student, University College Dublin (UCD)*  
Research conducted under the supervision of **Timilehin B. Aderinola** at **Insight SFI Centre for Data Analytics**.