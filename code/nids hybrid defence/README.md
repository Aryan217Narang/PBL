# NIDS Hybrid Adversarial Defense (CIC-DDoS2019)

Python implementation of the hybrid defense from Barik & Misra (2025):
PGD adversarial training (hyperparameters tuned via Pigeon-Inspired
Optimization) + Spatial Smoothing at test time, for a CNN-based NIDS.

## Setup
    pip install -r requirements.txt

Place CIC-DDoS2019 CSVs in `data/raw/`.

## Run the full 5-scenario pipeline
    python run_pipeline.py --data_glob "data/raw/*.csv" --label_col Label

This will:
1. Preprocess (MinMaxScaler -> ICA -> RFE)
2. Train a clean CNN baseline                  (Scenario 1)
3. Attack it with FGSM / JSMA / C&W             (Scenario 2)
4. Train a PGD-hardened CNN                     (Scenario 3)
5. Apply Spatial Smoothing only, at test time   (Scenario 4)
6. PIOA-tune PGD hyperparams, retrain + SS      (Scenario 5, proposed hybrid)

Results (accuracy/precision/recall/specificity/AUC/ASR) print to console and
save to `eval/results.json`.

## Layout
    data/preprocess.py         - load/clean/scale/ICA/RFE pipeline
    models/cnn.py               - 5-layer CNN classifier + training loop
    attacks/generate_attacks.py - FGSM/JSMA/C&W via ART
    defenses/pgd_training.py    - PGD adversarial training
    defenses/pioa.py            - Pigeon-Inspired Optimization for PGD hyperparams
    defenses/spatial_smoothing.py - test-time denoising filter
    eval/evaluate.py            - metrics (Acc/Prec/Rec/Spec/AUC/ASR) + scenario runner
    run_pipeline.py             - orchestrates all 5 scenarios end-to-end
