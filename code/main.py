"""
Hybrid Defense Model for DL-based NIDS Against Adversarial Attacks
Based on: Barik & Misra (2025), Multimedia Tools and Applications
Pipeline: Preprocessing → Attacks (JSMA/FGSM/C&W) → Defense (PGD+PIOA+SS) → Evaluation
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from consistency import detect_adversarial
from preprocessing import load_and_preprocess
from model import build_cnn_model
from attacks import generate_jsma, generate_fgsm, generate_cw
from defense import pgd_adversarial_training, spatial_smoothing, pioa_optimize
from evaluation import evaluate_model, print_results, plot_results

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Each dataset is now a LIST of individual CSV file paths
   # Change the "datasets" part in main.py to this:
"datasets": {
    "CIC-IDS2017": [
        "data/CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv",
        "data/CIC-IDS2017/Tuesday-WorkingHours.pcap_ISCX.csv",
        "data/CIC-IDS2017/Wednesday-workingHours.pcap_ISCX.csv",
        "data/CIC-IDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data/CIC-IDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "data/CIC-IDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "data/CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
},
    # Train/test split
    "test_size": 0.20,
    "val_split": 0.10,

    # CNN hyperparameters (from paper Table 3)
    "epochs":     10,
    "batch_size": 32,
    "optimizer":  "adam",
    "loss":       "sparse_categorical_crossentropy",

    # Attack parameter sets (from paper Table 4)
    "attack_parts": [
        {"max_iter": 10, "epsilon": 0.1 , "sigma": 1.5, "alpha": 0.003},
        {"max_iter": 15, "epsilon": 0.15, "sigma": 2.0, "alpha": 0.005},
        {"max_iter": 20, "epsilon": 0.2, "sigma": 2.5, "alpha": 0.007},
        {"max_iter": 25, "epsilon": 0.25, "sigma": 3.0, "alpha": 0.012},
    ],

    # PIOA hyperparameters (from paper Table 5)
    "pioa": {
        "n_pigeons":      50,
        "dimensions":     10,
        "max_iterations": 1000,
        "r":              0.5,
    },

    # PGD defense settings
    "pgd": {
        "epsilon": 0.02,
        "alpha":   0.003,
        "n_iter":  20,
    },

    # Output directory
    "output_dir": "results",
}


def run_pipeline(dataset_name: str, file_paths: list):
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*60}")

    out_dir = Path(CONFIG["output_dir"]) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. PREPROCESSING ────────────────────────────────────────
    print("\n[1/5] Preprocessing...")
    X_train, X_test, y_train, y_test, n_classes = load_and_preprocess(
        file_paths=file_paths,
        test_size=CONFIG["test_size"]
    )
    # 🔥 LIMIT DATA FOR SPEED (APPLY EVERYWHERE)
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    print(f"      Train: {X_train.shape} | Test: {X_test.shape} | Classes: {n_classes}")

    # ── 2. BUILD & TRAIN BASE MODEL ─────────────────────────────
    print("\n[2/5] Training base CNN model...")
    model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["val_split"],
        verbose=1
    )

    # ── Experiment 1: Pre-attack evaluation ─────────────────────
    print("\n── Experiment 1: Pre-attack Evaluation ──")
    pre_attack_results = evaluate_model(model, X_test, y_test, label="Pre-Attack")
    print_results(pre_attack_results)

    # ── 3. ADVERSARIAL ATTACKS ──────────────────────────────────
    print("\n[3/5] Generating adversarial attacks...")
    all_attack_results = {}

    for part_idx, params in enumerate(CONFIG["attack_parts"]):
        part_label = f"Part {part_idx + 1}"
        print(f"\n  Attack {part_label}: epsilon={params['epsilon']}, max_iter={params['max_iter']}")

        # JSMA
        print("    Generating JSMA...")
        X_jsma = generate_jsma(model, X_test, y_test, params)
        jsma_res = evaluate_model(model, X_jsma, y_test, label=f"Post-JSMA {part_label}")

        # FGSM
        print("    Generating FGSM...")
        X_fgsm = generate_fgsm(model, X_test, params["epsilon"])
        fgsm_res = evaluate_model(model, X_fgsm, y_test, label=f"Post-FGSM {part_label}")

        # C&W
        print("    Generating C&W...")
        X_cw = generate_cw(model, X_test, y_test, params)
        cw_res = evaluate_model(model, X_cw, y_test, label=f"Post-C&W {part_label}")

        all_attack_results[part_label] = {
            "jsma": jsma_res, "fgsm": fgsm_res, "cw": cw_res,
            "X_jsma": X_jsma, "X_fgsm": X_fgsm, "X_cw": X_cw, "params": params
        }
        print_results(jsma_res)
        print_results(fgsm_res)
        print_results(cw_res)

    # ── 4. DEFENSE ──────────────────────────────────────────────
    print("\n[4/5] Applying defense strategies...")
    defense_results = {}

    for part_idx, params in enumerate(CONFIG["attack_parts"]):
        part_label = f"Part {part_idx + 1}"
        X_jsma_test = all_attack_results[part_label]["X_jsma"]

        # ── Experiment 3: PGD single defense (training) ─────────
        print(f"\n  [{part_label}] PGD adversarial training...")
        pgd_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

        pgd_epsilon = pioa_optimize(
            base_epsilon=CONFIG["pgd"]["epsilon"],
            pioa_cfg=CONFIG["pioa"]
        )
        print(f"    PIOA-optimized epsilon: {pgd_epsilon:.4f}")

        X_train_adv, y_train_adv = pgd_adversarial_training(
            pgd_model, X_train, y_train,
            epsilon=pgd_epsilon,
            alpha=CONFIG["pgd"]["alpha"],
            n_iter=CONFIG["pgd"]["n_iter"]
        )

        pgd_model.fit(
            X_train_adv, y_train_adv,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            verbose=1
        )

        pgd_res = evaluate_model(pgd_model, X_jsma_test, y_test, label=f"PGD Defense {part_label}")
        print_results(pgd_res)

        # ── Experiment 4: SS single defense (testing) ───────────
        print(f"  [{part_label}] Spatial Smoothing defense...")
        X_test_smoothed = spatial_smoothing(X_jsma_test, window_radius=params["sigma"])
        ss_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)
        ss_model.fit(
            X_train, y_train,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            verbose=1
        )
        ss_res = evaluate_model(ss_model, X_test_smoothed, y_test, label=f"SS Defense {part_label}")
        print_results(ss_res)

        # ── Experiment 5: Hybrid defense (PGD+PIOA train + SS test)
        print(f"  [{part_label}] Hybrid defense (PGD+PIOA+SS)...")
        hybrid_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

        hybrid_model.fit(
            X_train_adv, y_train_adv,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            verbose=1
        )

        X_hybrid_test = spatial_smoothing(X_jsma_test, window_radius=params["sigma"])
        # ── YOUR NEW CONSISTENCY DETECTION ─────────────
        print(f"  [{part_label}] Applying AICC + TCC detection...")
        #  LIMIT DATA FOR SPEED (IMPORTANT)
        X_hybrid_test = X_hybrid_test[:1000]
        y_test = y_test[:1000]
        adv_flags = detect_adversarial(
            hybrid_model,
            X_hybrid_test,
            window_size=3,
            aicc_thresh=0.7,
            final_thresh=0.75
        )
        from evaluation import evaluate_detection

        # For adversarial test data → all labels = 1
        y_adv_true = np.ones(len(adv_flags))

        det_res = evaluate_detection(adv_flags, y_adv_true)

        print("\n    ── Detection Metrics ──")
        print(f"    Detection Rate : {det_res['detection_rate']}%")
        print(f"    False Alarm Rate : {det_res['false_alarm_rate']}%")

        # Optional: filter detected adversarial samples
        clean_indices = np.where(adv_flags == 0)[0]

        X_filtered = X_hybrid_test[clean_indices]
        y_filtered = y_test[clean_indices]

        print(f"    Removed {len(X_hybrid_test) - len(X_filtered)} suspected adversarial samples")

        # Evaluate on filtered data
        hybrid_res = evaluate_model(
            hybrid_model,
            X_filtered,
            y_filtered,
            label=f"Hybrid + AICC+TCC {part_label}"
        )


        print_results(hybrid_res)

        defense_results[part_label] = {
            "pgd": pgd_res,
            "ss":  ss_res,
            "hybrid": hybrid_res,
        }

    # ── 5. PLOT & SAVE ──────────────────────────────────────────
    print(f"\n[5/5] Saving results to {out_dir}/")
    plot_results(
        pre_attack_results,
        all_attack_results,
        defense_results,
        history,
        out_dir=str(out_dir)
    )
    print(f"      Done. Results saved to: {out_dir}/")
    return defense_results


def main():
    Path(CONFIG["output_dir"]).mkdir(exist_ok=True)

    for name, file_paths in CONFIG["datasets"].items():
        # Check which files actually exist
        missing = [f for f in file_paths if not Path(f).exists()]
        if missing:
            print(f"\n[SKIP] {name}: the following files were not found:")
            for m in missing:
                print(f"       {m}")
            continue
        run_pipeline(name, file_paths)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()