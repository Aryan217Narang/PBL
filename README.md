# 🛡️ Hybrid Adversarial Defense for Deep Learning-Based NIDS

A comprehensive defense framework for Deep Learning-based Network Intrusion Detection Systems (NIDS) against adversarial evasion attacks on the **CIC-DDoS2019** benchmark dataset, reproducing and extending the methodology by **Barik & Misra (2025)** (*Multimedia Tools and Applications*).

---

## ⚡ Canonical Pipeline Architecture

> **Notice:** The repository has been standardized on a **single, unified, GPU-accelerated PyTorch + ART pipeline** located in `code/nids hybrid defence/`.  
> The legacy TensorFlow/Keras CPU prototype has been **deprecated** and moved to `code/legacy_tf_pipeline/`.

```
                                  [ Network Flow Traffic ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ Preprocessing & Feature Eng.  │
                             │ (MinMaxScaler -> ICA -> RFE)  │
                             └───────────────┬───────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
            [ Training Phase ]                              [ Testing Phase ]
                      │                                             │
      ┌───────────────┴───────────────┐                             ▼
      │ PGD Adversarial Training      │              ┌─────────────────────────────┐
      │  + PIOA Optimization          │              │ Adversarial Attack Suite    │
      │  (Map & Compass + Landmark)   │              │ (FGSM, JSMA, C&W L2)        │
      └───────────────┬───────────────┘              └──────────────┬──────────────┘
                      │                                             │
                      │ Hardened Model Weights                      │
                      ▼                                             ▼
      ┌───────────────────────────────┐              ┌─────────────────────────────┐
      │      Trained NIDS CNN         │ ◄─────────── │ Spatial Smoothing (SS)      │
      │   (5-Layer 1D ConvNet)        │              │ + AICC/TCC Consistency Chk  │
      └───────────────┬───────────────┘              └─────────────────────────────┘
                      │
                      ▼
        [ Final Robust Classification ]
```

---

## 🔒 Data Leakage Auditing & Fixes Applied

In this release, all data and label leakage vectors have been identified and strictly eliminated:

1. **Elimination of Target / Label Leakage**:
   - *Previous bug:* If the consistency detector flagged an anomaly (`adv_flags[i] == 1`), legacy evaluators inspected ground truth `y[i]` and assigned `preds[i] = y[i]`.
   - *Fix:* Replaced with an **active NIDS quarantine policy**: flagged flows are autonomously quarantined as **Malicious / Attack (`preds[i] = 1`)**. Zero ground-truth test labels are accessed during inference. Any false alarm on legitimate benign traffic honestly penalizes Specificity and Precision.

2. **Elimination of Transductive Batch Leakage**:
   - *Previous bug:* Threshold adaptation computed `np.mean(c_final)` across the entire test set simultaneously, allowing future test samples to influence earlier decisions.
   - *Fix:* Threshold calibration is now strictly **causal** (using an online running average of past observed flows $0 \dots i$) or fixed to a pre-calibrated validation threshold. Future samples never leak into current decisions.

3. **Causal Chronological Evaluation for TCC**:
   - *Previous bug:* Shuffling test data randomized packet order, breaking temporal correlation for Temporal Consistency Checking (TCC).
   - *Fix:* Added continuous chronological stream evaluation (`--temporal`) using real microsecond timestamps from the raw capture files.

4. **Honest Formulaic ASR (Attack Success Rate)**:
   - *Previous bug:* Hardcoded `0.0%` ASR whenever consistency was enabled.
   - *Fix:* Formulaic measurement of true evasion rate: fraction of attacks evading both the CNN and the consistency quarantine.

---

## 📊 Benchmark Results across Scenarios

Evaluation conducted on the **CIC-DDoS2019** benchmark with GPU acceleration (NVIDIA CUDA):

| # | Scenario | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | AUC | ASR (%) | False Alarm (FAR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Pre-Attack Baseline (Clean CNN)** | **99.13%** | **98.97%** | **99.12%** | **99.15%** | **0.9990** | **0.00%** | *n/a* |
| **2** | **No Defense vs FGSM** ($\epsilon=0.10$) | 93.80% | 90.20% | 81.42% | 97.42% | 0.9911 | 18.92% | *n/a* |
| **2** | **No Defense vs JSMA** ($\theta=0.2$) | 70.60% | 33.65% | 30.97% | 82.17% | 0.8440 | 70.27% | *n/a* |
| **2** | **No Defense vs C&W** ($L_2$) | 72.40% | 33.33% | 22.12% | 87.08% | 0.8937 | 79.28% | *n/a* |
| **3** | **PGD-only Defense vs FGSM** | 99.20% | 96.58% | 100.00% | 98.97% | 0.9997 | 0.00% | *n/a* |
| **3** | **PGD-only Defense vs JSMA** | 95.20% | 85.04% | 95.58% | 95.09% | 0.9856 | 04.50% | *n/a* |
| **3** | **PGD-only Defense vs C&W** | 97.40% | 93.86% | 94.69% | 98.19% | 0.9940 | 05.41% | *n/a* |
| **4** | **SS-only Defense ($w=3$) vs FGSM** | 39.60% | 26.08% | 91.15% | 24.55% | 0.6593 | 09.01% | *n/a* |
| **4** | **SS-only Defense ($w=3$) vs JSMA** | 28.20% | 20.29% | 74.34% | 14.73% | 0.5483 | 26.13% | *n/a* |
| **4** | **SS-only Defense ($w=3$) vs C&W** | 42.40% | 26.29% | 85.84% | 29.72% | 0.6375 | 14.41% | *n/a* |
| **5a** | **Proposed Hybrid (PGD+PIOA+SS) vs FGSM** | 75.40% | 47.28% | 76.99% | 74.94% | 0.8503 | 23.42% | *n/a* |
| **5b** | **Hybrid + AICC vs FGSM** | 74.80% | 46.52% | 76.99% | 74.16% | 0.8503 | 22.52% | 1.80% |
| **5c** | **Hybrid + TCC vs FGSM** | 75.20% | 47.03% | 76.99% | 74.68% | 0.8503 | 23.42% | 0.80% |
| **5d** | **Hybrid + AICC & TCC vs FGSM** | 75.00% | 46.77% | 76.99% | 74.42% | 0.8503 | 23.42% | 1.20% |
| **5a** | **Proposed Hybrid (PGD+PIOA+SS) vs JSMA** | 70.60% | 41.67% | 75.22% | 69.25% | 0.7996 | 25.23% | *n/a* |
| **5d** | **Hybrid + AICC & TCC vs JSMA** | 70.00% | 41.15% | 76.11% | 68.22% | 0.7996 | 24.32% | 0.60% |
| **5a** | **Proposed Hybrid (PGD+PIOA+SS) vs C&W** | 78.20% | 51.20% | 75.22% | 79.07% | 0.8437 | 25.23% | *n/a* |
| **5d** | **Hybrid + AICC & TCC vs C&W** | 77.60% | 50.30% | 75.22% | 78.29% | 0.8437 | 25.23% | 0.40% |

---

## 🚀 How to Run the Canonical Pipeline

### 1. Unified Entry Point (Root)
To run the full end-to-end evaluation:
```bash
python code/main.py
```

### 2. Chronological Temporal Stream Benchmark
To evaluate on 1,500 continuous flows with real timestamps:
```bash
python code/main.py --temporal --attack_subset 1500
```

### 3. Direct PyTorch Pipeline Commands
Alternatively, execute directly from within `code/nids hybrid defence/`:
```bash
cd "code/nids hybrid defence"

# Standard evaluation
python run_pipeline.py

# Temporal stream evaluation with custom subset
python run_pipeline.py --temporal --attack_subset 1500

# Retrain all checkpoints from scratch
python run_pipeline.py --retrain --epochs 6
```

---

## 📁 Repository Structure

```
├── code/
│   ├── main.py                          # Unified CLI entry point
│   ├── legacy_tf_pipeline/              # DEPRECATED: TensorFlow prototype
│   │   ├── README.md                    # Deprecation documentation
│   │   ├── main_tf_legacy.py
│   │   ├── preprocessing.py
│   │   └── ...
│   │
│   └── nids hybrid defence/             # CANONICAL UNIFIED PIPELINE (PyTorch)
│       ├── run_pipeline.py              # Main execution orchestrator
│       ├── data/
│       │   ├── preprocess.py            # Leak-free split & feature engineering
│       │   └── processed/               # Cached processed npz and pipeline joblib
│       ├── models/
│       │   ├── cnn.py                   # 5-layer 1D CNN architecture
│       │   └── checkpoints/             # Trained weights (cnn_clean, cnn_pgd, cnn_hybrid)
│       ├── attacks/
│       │   └── generate_attacks.py      # ART attack suite (FGSM, JSMA, C&W L2)
│       ├── defenses/
│       │   ├── pgd_training.py          # Multi-scale PGD adversarial hardening
│       │   ├── pioa.py                  # Pigeon-Inspired Optimization algorithm
│       │   ├── spatial_smoothing.py     # Inference-time median filter
│       │   └── consistency.py           # Causal AICC & TCC anomaly detectors
│       └── eval/
│           ├── evaluate.py              # Honest active NIDS evaluation (no label snooping)
│           ├── results.json             # Full metric JSON dump
│           └── summary_results.csv      # Formatted summary matrix
│
├── CIC_DDoS2019_Hybrid_Defense_Changes_Report.pdf  # Peer review report document
└── README.md                            # Project documentation
```
