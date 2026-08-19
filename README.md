# 🛡️ Hybrid Adversarial Defense for Deep Learning-Based NIDS

A comprehensive defense framework for Deep Learning-based Network Intrusion Detection Systems (NIDS) against adversarial evasion attacks on the **CIC-DDoS2019** benchmark dataset, reproducing and extending the methodology by **Barik & Misra (2025)** (*Multimedia Tools and Applications*).

---

## 📌 Architecture Overview

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

## 📊 Experimental Results across 5 Scenarios

Evaluation conducted on the **CIC-DDoS2019** benchmark (150,000 network flows across 10 DDoS categories) with GPU acceleration (NVIDIA RTX 4060):

| # | Scenario | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | AUC | ASR (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Pre-Attack Baseline (Clean CNN)** | **99.97%** | **99.97%** | **99.99%** | **99.74%** | **1.0000** | *n/a* |
| **2** | **Post-Attack (No Defense vs FGSM)** | 78.40% | 89.15% | 86.35% | 11.32% | 0.4727 | 21.60% |
| **2** | **Post-Attack (No Defense vs JSMA)** | 72.40% | 89.51% | 78.30% | 22.64% | 0.4930 | 27.60% |
| **2** | **Post-Attack (No Defense vs C&W)** | 80.60% | 90.14% | 87.92% | 18.87% | 0.5122 | 19.40% |
| **3** | **Single Defense (PGD-Only vs FGSM)** | 78.60% | 89.17% | 86.58% | 11.32% | 0.5055 | 21.40% |
| **3** | **Single Defense (PGD-Only vs JSMA)** | 73.60% | 88.70% | 80.76% | 13.21% | 0.4711 | 26.40% |
| **3** | **Single Defense (PGD-Only vs C&W)** | 78.20% | 89.30% | 85.91% | 13.21% | 0.5155 | 21.80% |
| **4** | **Single Defense (SS-Only vs FGSM)** | 83.40% | 89.57% | 92.17% | 09.43% | 0.5089 | *n/a* |
| **4** | **Single Defense (SS-Only vs JSMA)** | 76.40% | 89.07% | 83.89% | 13.21% | 0.5052 | *n/a* |
| **4** | **Single Defense (SS-Only vs C&W)** | 83.60% | 89.42% | 92.62% | 07.55% | 0.5249 | *n/a* |
| **5** | **Proposed Hybrid (PGD+PIOA+SS) vs FGSM** | **74.80%** | **89.63%** | **81.21%** | **20.75%** | **0.4827** | **21.40%** |
| **5** | **Proposed Hybrid (PGD+PIOA+SS) vs JSMA** | **60.60%** | **89.31%** | **63.53%** | **35.85%** | **0.4766** | **33.80%** |
| **5** | **Proposed Hybrid (PGD+PIOA+SS) vs C&W** | **80.20%** | **89.55%** | **88.14%** | **13.21%** | **0.5242** | **24.00%** |

---

## 📈 Multi-Part Parameter Benchmark (CIC-DDoS2019)

Detailed metrics across perturbation parameters ($\epsilon = 0.10 \rightarrow 0.25$, $\sigma = 1.5 \rightarrow 3.0$):

| Experiment / Scenario | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC | Specificity (%) | ASR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-Attack Clean** | **98.40%** | **99.87%** | **98.03%** | **98.94%** | **0.9995** | **98.80%** | *n/a* |
| Post-JSMA (Part 1: $\epsilon=0.10$) | 86.70% | 95.50% | 86.58% | 90.82% | 0.9451 | 86.83% | 12.60% |
| Post-FGSM (Part 1: $\epsilon=0.10$) | 75.90% | 75.98% | 99.87% | 86.30% | 0.5034 | 49.93% | 24.39% |
| Post-C&W (Part 1: $c=0.1$) | 76.00% | 76.00% | 100.00% | 86.36% | 0.5000 | 50.00% | 24.29% |
| **PGD Defense (Part 1)** | **93.20%** | **97.01%** | **93.95%** | **95.45%** | **0.9781** | **92.39%** | **5.39%** |
| **SS Defense (Part 1)** | **80.30%** | **80.97%** | **96.84%** | **88.20%** | **0.8423** | **62.38%** | **19.15%** |
| **Hybrid + AICC/TCC (Part 1)** | **71.70%** | **91.33%** | **69.34%** | **78.83%** | **0.8096** | **74.25%** | **0.0%** |
| Post-JSMA (Part 2: $\epsilon=0.15$) | 88.10% | 92.23% | 92.11% | 92.17% | 0.8850 | 83.76% | 11.18% |
| Post-FGSM (Part 2: $\epsilon=0.15$) | 75.80% | 75.95% | 99.74% | 86.23% | 0.4891 | 49.87% | 24.49% |
| Post-C&W (Part 2: $c=0.15$) | 76.00% | 76.00% | 100.00% | 86.36% | 0.5000 | 50.00% | 24.29% |
| **PGD Defense (Part 2)** | **75.20%** | **98.48%** | **68.42%** | **80.75%** | **0.9478** | **82.54%** | **22.75%** |
| **SS Defense (Part 2)** | **75.20%** | **82.41%** | **85.66%** | **84.00%** | **0.6984** | **63.87%** | **24.16%** |
| **Hybrid + AICC/TCC (Part 2)** | **78.20%** | **93.15%** | **76.97%** | **84.29%** | **0.8251** | **79.53%** | **0.0%** |
| Post-JSMA (Part 3: $\epsilon=0.20$) | 84.90% | 87.27% | 93.82% | 90.42% | 0.8650 | 75.24% | 14.74% |
| Post-FGSM (Part 3: $\epsilon=0.20$) | 75.90% | 75.98% | 99.87% | 86.30% | 0.4946 | 49.93% | 24.39% |
| Post-C&W (Part 3: $c=0.20$) | 76.00% | 76.00% | 100.00% | 86.36% | 0.5000 | 50.00% | 24.29% |
| **PGD Defense (Part 3)** | **56.20%** | **93.05%** | **45.79%** | **61.38%** | **0.8019** | **67.48%** | **43.33%** |
| **SS Defense (Part 3)** | **77.10%** | **77.92%** | **97.50%** | **86.62%** | **0.6622** | **55.00%** | **22.49%** |
| **Hybrid + AICC/TCC (Part 3)** | **61.30%** | **85.39%** | **59.21%** | **69.93%** | **0.7236** | **63.56%** | **0.0%** |
| Post-JSMA (Part 4: $\epsilon=0.25$) | 86.50% | 87.97% | 95.26% | 91.47% | 0.8819 | 77.01% | 13.11% |
| Post-FGSM (Part 4: $\epsilon=0.25$) | 75.90% | 75.98% | 99.87% | 86.30% | 0.5003 | 49.93% | 24.39% |
| Post-C&W (Part 4: $c=0.25$) | 76.00% | 76.00% | 100.00% | 86.36% | 0.5000 | 50.00% | 24.29% |
| **PGD Defense (Part 4)** | **45.70%** | **95.78%** | **29.87%** | **45.54%** | **0.8535** | **62.85%** | **53.66%** |
| **SS Defense (Part 4)** | **77.70%** | **79.80%** | **94.61%** | **86.57%** | **0.7197** | **59.39%** | **22.08%** |
| **Hybrid + AICC/TCC (Part 4)** | **39.10%** | **98.09%** | **20.26%** | **33.59%** | **0.7602** | **59.51%** | **0.0%** |

---

## 🛠️ Key Components

1. **Preprocessing Pipeline**:
   - `MinMaxScaler` normalization to $[0, 1]$.
   - `FastICA` feature unmixing & decorrelation.
   - `Recursive Feature Elimination (RFE)` selecting the top discriminative features.
2. **Classifier**:
   - 5-layer 1D Convolutional Neural Network (`Conv1D`, `BatchNorm1D`, `MaxPool1D`, `Dropout`, `Dense`).
3. **Adversarial Attacks**:
   - **FGSM** (Fast Gradient Sign Method)
   - **JSMA** (Jacobian Saliency Map Attack)
   - **C&W** (Carlini & Wagner $L_2$)
4. **Hybrid Defense Architecture**:
   - **Training-Phase**: Projected Gradient Descent (PGD) hardening optimized by Pigeon-Inspired Optimization Algorithm (**PIOA**).
   - **Testing-Phase**: Spatial Smoothing (**SS**) median filtering + Adaptive Input Consistency Check (**AICC**) & Temporal Consistency Check (**TCC**).

---

## 🚀 Setup & Execution

### 1. Installation
```bash
git clone https://github.com/Aryan217Narang/PBL.git
cd PBL/code/"nids hybrid defence"
pip install -r requirements.txt
```

### 2. Run the 5-Scenario Pipeline via CLI
```bash
python run_pipeline.py --epochs 8 --attack_subset 500 --pioa_pigeons 5 --pioa_iters 2
```

### 3. Run the Interactive Streamlit Web UI
```bash
cd PBL/code
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── README.md
├── .gitignore
└── code/
    ├── app.py                          # Interactive Streamlit Web UI
    ├── preprocessing.py                # Preprocessing module (MinMaxScaler, ICA, RFE)
    ├── model.py                        # 1D-CNN classifier definition
    ├── attacks.py                      # Vectorized adversarial attacks (FGSM, JSMA, C&W)
    ├── defense.py                      # PGD, Spatial Smoothing, and PIOA
    ├── consistency.py                  # AICC + TCC consistency detector
    ├── evaluation.py                   # Multi-metric evaluation suite
    ├── main.py                         # TensorFlow/Keras pipeline runner
    ├── results/
    │   └── CIC-DDoS2019/               # Confusion matrices, ROC curves & summary CSV
    └── nids hybrid defence/            # PyTorch + ART GPU Modular Pipeline
        ├── README.md
        ├── requirements.txt
        ├── run_pipeline.py             # 5-Scenario Orchestrator
        ├── data/
        │   └── preprocess.py
        ├── models/
        │   ├── cnn.py
        │   └── checkpoints/
        │       ├── cnn_clean.pt
        │       ├── cnn_pgd.pt
        │       └── cnn_hybrid.pt
        ├── attacks/
        │   └── generate_attacks.py
        ├── defenses/
        │   ├── pgd_training.py
        │   ├── pioa.py
        │   └── spatial_smoothing.py
        └── eval/
            ├── evaluate.py
            └── results.json
```

---

## 📚 References
* **Barik, K., & Misra, S. (2025).** *A comprehensive defense approach of deep learning-based NIDS against adversarial attacks.* Multimedia Tools and Applications, 84:37745–37791. https://doi.org/10.1007/s11042-025-21008-5
