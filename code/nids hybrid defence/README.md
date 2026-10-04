# NIDS Hybrid Adversarial Defense (CIC-DDoS2019)

Python implementation of the hybrid defense from **Barik & Misra (2025)** (*Multimedia Tools and Applications*):
PGD adversarial training (hyperparameters tuned via Pigeon-Inspired Optimization) + Spatial Smoothing at test time + Adaptive Input & Temporal Consistency Checking (AICC/TCC), for a CNN-based NIDS.

---

## 📊 Benchmark Results (5 Scenarios)

| # | Scenario | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | AUC | ASR (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Clean CNN Baseline** | **99.97%** | **99.97%** | **99.99%** | **99.74%** | **1.0000** | *n/a* |
| **2** | **Post-Attack (No Defense vs FGSM)** | 78.40% | 89.15% | 86.35% | 11.32% | 0.4727 | 21.60% |
| **2** | **Post-Attack (No Defense vs JSMA)** | 72.40% | 89.51% | 78.30% | 22.64% | 0.4930 | 27.60% |
| **2** | **Post-Attack (No Defense vs C&W)** | 80.60% | 90.14% | 87.92% | 18.87% | 0.5122 | 19.40% |
| **3** | **Single Defense (PGD-Only vs FGSM)** | 78.60% | 89.17% | 86.58% | 11.32% | 0.5055 | 21.40% |
| **3** | **Single Defense (PGD-Only vs JSMA)** | 73.60% | 88.70% | 80.76% | 13.21% | 0.4711 | 26.40% |
| **3** | **Single Defense (PGD-Only vs C&W)** | 78.20% | 89.30% | 85.91% | 13.21% | 0.5155 | 21.80% |
| **4** | **Single Defense (SS-Only vs FGSM)** | 83.40% | 89.57% | 92.17% | 09.43% | 0.5089 | *n/a* |
| **4** | **Single Defense (SS-Only vs JSMA)** | 76.40% | 89.07% | 83.89% | 13.21% | 0.5052 | *n/a* |
| **4** | **Single Defense (SS-Only vs C&W)** | 83.60% | 89.42% | 92.62% | 07.55% | 0.5249 | *n/a* |
| **5** | **Proposed Hybrid (PGD+PIOA+SS+AICC/TCC) vs FGSM** | **76.00%** | **80.85%** | **92.68%** | **00.00%** | **0.5488** | **0.00%** |
| **5** | **Proposed Hybrid (PGD+PIOA+SS+AICC/TCC) vs JSMA** | **66.00%** | **81.58%** | **75.61%** | **22.22%** | **0.4553** | **0.00%** |
| **5** | **Proposed Hybrid (PGD+PIOA+SS+AICC/TCC) vs C&W** | **70.00%** | **80.95%** | **82.93%** | **11.11%** | **0.5068** | **0.00%** |

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

Place CIC-DDoS2019 CSVs in `data/raw/` (or let the pipeline synthesize standard benchmark traffic automatically if running without dataset files).

---

## 🚀 Run the 5-Scenario Pipeline

```bash
python run_pipeline.py --epochs 8 --attack_subset 500 --pioa_pigeons 5 --pioa_iters 2
```

This executes:
1. **Preprocessing**: MinMaxScaler $\rightarrow$ FastICA $\rightarrow$ RFE feature extraction.
2. **Scenario 1**: Clean 5-layer 1D CNN baseline training.
3. **Scenario 2**: Attack generation and evaluation (FGSM, JSMA, C&W $L_2$).
4. **Scenario 3**: Multi-scale PGD adversarial training.
5. **Scenario 4**: Test-time median Spatial Smoothing (SS).
6. **Scenario 5**: Proposed Hybrid Defense (PIOA-optimized PGD + SS + AICC/TCC consistency checking).

Results (Accuracy, Precision, Recall, Specificity, AUC, ASR) print to console and persist in `eval/results.json`.

---

## 📁 Layout

```
    data/preprocess.py            - load/clean/scale/ICA/RFE pipeline
    models/cnn.py                 - 5-layer 1D CNN classifier + training loop
    attacks/generate_attacks.py   - FGSM / JSMA / C&W via ART (Adversarial Robustness Toolbox)
    defenses/pgd_training.py      - multi-scale PGD adversarial training
    defenses/pioa.py              - Pigeon-Inspired Optimization for PGD hyperparams
    defenses/spatial_smoothing.py - test-time median filtering
    defenses/consistency.py       - Adaptive Input & Temporal Consistency Check (AICC/TCC)
    eval/evaluate.py              - metrics (Acc/Prec/Rec/Spec/AUC/ASR) + scenario runner
    eval/results.json             - benchmark metrics persistence
    run_pipeline.py               - orchestrates all 5 scenarios end-to-end
```
