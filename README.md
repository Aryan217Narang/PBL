# 🛡️ Hybrid Adversarial Defense for Deep Learning-Based NIDS

A comprehensive defense framework for Deep Learning-based Network Intrusion Detection Systems (NIDS) against adversarial evasion attacks, reproducing and extending the methodology by **Barik & Misra (2025)** (*Multimedia Tools and Applications*).

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

Evaluation conducted on the **CIC-DDoS2019 / CIC-IDS2017** benchmark with GPU acceleration (NVIDIA RTX 4060):

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

## 🛠️ Key Components

1. **Preprocessing Pipeline**:
   - `MinMaxScaler` normalization to $[0, 1]$.
   - `FastICA` feature unmixing & decorrelation.
   - `Recursive Feature Elimination (RFE)` selecting the top 20 discriminative features.
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
python run_pipeline.py --epochs 10 --attack_subset 500 --pioa_pigeons 5 --pioa_iters 3
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
