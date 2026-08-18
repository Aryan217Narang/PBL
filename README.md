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
| **Pre-Attack Clean** | **82.30%** | **76.70%** | **82.30%** | **78.50%** | **0.9813** | **98.20%** | *n/a* |
| Post-JSMA (Part 1: $\epsilon=0.10$) | 68.00% | 60.93% | 68.00% | 63.45% | 0.9125 | 96.58% | *n/a* |
| Post-FGSM (Part 1: $\epsilon=0.10$) | 13.20% | 12.62% | 13.20% | 11.80% | 0.4890 | 90.81% | *n/a* |
| Post-C&W (Part 1: $c=0.1$) | 6.00% | 0.36% | 6.00% | 0.68% | 0.5024 | 90.91% | *n/a* |
| **PGD Defense (Part 1)** | **72.90%** | **66.04%** | **72.90%** | **67.63%** | **0.9423** | **97.19%** | *n/a* |
| **SS Defense (Part 1)** | **22.70%** | **17.41%** | **22.70%** | **16.83%** | **0.5609** | **91.33%** | *n/a* |
| **Hybrid + AICC/TCC (Part 1)** | **56.80%** | **71.58%** | **56.80%** | **53.37%** | **0.7105** | **95.71%** | **0.0%** |
| Post-JSMA (Part 2: $\epsilon=0.15$) | 62.40% | 55.00% | 62.40% | 56.02% | 0.7953 | 95.65% | *n/a* |
| **PGD Defense (Part 2)** | **47.30%** | **41.57%** | **47.30%** | **39.00%** | **0.8136** | **93.80%** | *n/a* |
| **SS Defense (Part 2)** | **24.40%** | **24.81%** | **24.40%** | **23.06%** | **0.6368** | **91.87%** | *n/a* |
| **Hybrid + AICC/TCC (Part 2)** | **56.30%** | **67.07%** | **56.30%** | **53.79%** | **0.6983** | **94.95%** | **0.0%** |
| Post-JSMA (Part 3: $\epsilon=0.20$) | 45.60% | 52.75% | 45.60% | 39.31% | 0.7661 | 93.57% | *n/a* |
| **PGD Defense (Part 3)** | **45.30%** | **35.17%** | **45.30%** | **38.39%** | **0.8081** | **93.65%** | *n/a* |
| **SS Defense (Part 3)** | **38.60%** | **33.43%** | **38.60%** | **32.37%** | **0.6566** | **93.09%** | *n/a* |
| **Hybrid + AICC/TCC (Part 3)** | **53.20%** | **63.00%** | **53.20%** | **54.60%** | **0.6111** | **94.63%** | **0.0%** |
| Post-JSMA (Part 4: $\epsilon=0.25$) | 60.30% | 61.68% | 60.30% | 54.81% | 0.8266 | 95.46% | *n/a* |
| **PGD Defense (Part 4)** | **46.80%** | **35.42%** | **46.80%** | **37.18%** | **0.7285** | **93.70%** | *n/a* |
| **SS Defense (Part 4)** | **35.80%** | **29.50%** | **35.80%** | **30.55%** | **0.6597** | **92.63%** | *n/a* |
| **Hybrid + AICC/TCC (Part 4)** | **61.30%** | **61.90%** | **61.30%** | **59.66%** | **0.6728** | **95.58%** | **0.0%** |

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
