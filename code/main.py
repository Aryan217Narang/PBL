"""
main.py
Execution pipeline for NIDS Adversarial Defense framework (CIC-DDoS2019):
  1. Preprocessing (MinMaxScaler -> ICA -> RFE -> Binary Intrusion Split)
  2. Baseline Model Training (1D-CNN)
  3. Adversarial Attacks (FGSM, JSMA, C&W L2)
  4. Defenses (PGD Adversarial Training + PIOA Optimization + Spatial Smoothing + AICC/TCC Consistency Detection)
  5. Evaluation, Plots & CSV Metric Reports
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from pathlib import Path

from consistency import detect_adversarial
from preprocessing import load_and_preprocess
from model import build_cnn_model
from attacks import generate_jsma, generate_fgsm, generate_cw
from defense import pgd_adversarial_training, spatial_smoothing, pioa_optimize
from evaluation import evaluate_model, evaluate_hybrid_defense, print_results, plot_results


# ─────────────────────────────────────────────
# Path Resolver
# ─────────────────────────────────────────────
def resolve_path(p: str) -> str:
    path_obj = Path(p)
    if path_obj.exists():
        return str(path_obj)
    cand1 = Path(__file__).resolve().parent / p
    if cand1.exists():
        return str(cand1)
    cand2 = Path(__file__).resolve().parent.parent / p
    if cand2.exists():
        return str(cand2)
    return str(path_obj)


# ─────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "datasets": {
        "CIC-DDoS2019": [
            "data/cic-ids-2019/DrDoS_DNS_data_1_per.csv",
            "data/cic-ids-2019/DrDoS_LDAP_data_2_0_per.csv",
            "data/cic-ids-2019/DrDoS_MSSQL_data_1_3_per.csv",
            "data/cic-ids-2019/DrDoS_NTP_data_data_5_per.csv",
            "data/cic-ids-2019/DrDoS_NetBIOS_data_1_3_per.csv",
            "data/cic-ids-2019/DrDoS_SNMP_data_1_3_per.csv",
            "data/cic-ids-2019/DrDoS_SSDP_data_2_per.csv",
            "data/cic-ids-2019/DrDoS_UDP_data_2_per.csv",
            "data/cic-ids-2019/UDPLag_data_2_0_per.csv",
            "data/cic-ids-2019/syn_data.csv"
        ]
    },
    "test_size": 0.20,
    "batch_size": 128,
    "epochs": 10,
    "val_split": 0.10,

    "attack_parts": [
        {"max_iter": 10, "epsilon": 0.10, "sigma": 1.5, "alpha": 0.003},
        {"max_iter": 15, "epsilon": 0.15, "sigma": 2.0, "alpha": 0.005},
        {"max_iter": 20, "epsilon": 0.20, "sigma": 2.5, "alpha": 0.007},
        {"max_iter": 25, "epsilon": 0.25, "sigma": 3.0, "alpha": 0.012},
    ],

    "pgd": {
        "epsilon": 0.02,
        "alpha":   0.003,
        "n_iter":  10,
    },

    "pioa": {
        "n_pigeons":      20,
        "dimensions":     5,
        "max_iterations": 20,
        "r":              0.5,
    },

    "output_dir": "results",
}


def run_pipeline(dataset_name: str, raw_file_paths: list):
    print(f"\n{'='*65}")
    print(f"  DATASET: {dataset_name} (100% 2019 NIDS Dataset)")
    print(f"{'='*65}")

    file_paths = [resolve_path(f) for f in raw_file_paths]

    out_dir = Path(CONFIG["output_dir"]) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. PREPROCESSING ────────────────────────────────────────
    print("\n[1/5] Preprocessing (MinMaxScaler -> FastICA -> RFE (25 Features) -> Binary Intrusion Split)...")
    X_train, X_test, y_train, y_test, n_classes, class_weights = load_and_preprocess(
        file_paths=file_paths,
        test_size=CONFIG["test_size"],
        n_ica_components=30,
        n_rfe_features=25,
        force_recompute=True
    )
    # Test subset for rapid adversarial generation
    X_test_sub = X_test[:1000]
    y_test_sub = y_test[:1000]
    print(f"      Train: {X_train.shape} | Test: {X_test_sub.shape} | Classes: {n_classes}")

    # ── 2. BUILD & TRAIN BASE MODEL ─────────────────────────────
    print("\n[2/5] Training base 1D-CNN classifier with class weighting...")
    model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["val_split"],
        class_weight=class_weights,
        verbose=1
    )

    # ── Experiment 1: Pre-attack evaluation ─────────────────────
    print("\n-- Experiment 1: Pre-attack Baseline Evaluation --")
    pre_attack_results = evaluate_model(model, X_test_sub, y_test_sub, label="Pre-Attack Clean")
    print_results(pre_attack_results)

    # ── 3. ADVERSARIAL ATTACKS ──────────────────────────────────
    print("\n[3/5] Generating adversarial attacks (FGSM, JSMA, C&W)...")
    all_attack_results = {}

    for part_idx, params in enumerate(CONFIG["attack_parts"]):
        part_label = f"Part {part_idx + 1}"
        print(f"\n  Attack {part_label}: epsilon={params['epsilon']}, max_iter={params['max_iter']}")

        # JSMA
        print("    Generating JSMA...")
        X_jsma = generate_jsma(model, X_test_sub, y_test_sub, params)
        jsma_res = evaluate_model(model, X_jsma, y_test_sub, label=f"Post-JSMA {part_label}", X_clean=X_test_sub)

        # FGSM
        print("    Generating FGSM...")
        X_fgsm = generate_fgsm(model, X_test_sub, params["epsilon"])
        fgsm_res = evaluate_model(model, X_fgsm, y_test_sub, label=f"Post-FGSM {part_label}", X_clean=X_test_sub)

        # C&W
        print("    Generating C&W...")
        X_cw = generate_cw(model, X_test_sub, y_test_sub, params)
        cw_res = evaluate_model(model, X_cw, y_test_sub, label=f"Post-C&W {part_label}", X_clean=X_test_sub)

        all_attack_results[part_label] = {
            "jsma": jsma_res, "fgsm": fgsm_res, "cw": cw_res,
            "X_jsma": X_jsma, "X_fgsm": X_fgsm, "X_cw": X_cw, "params": params
        }
        print_results(jsma_res)
        print_results(fgsm_res)
        print_results(cw_res)

    # ── 4. DEFENSE STRATEGIES ───────────────────────────────────
    print("\n[4/5] Applying Defense Strategies (PGD, Spatial Smoothing, PIOA Hybrid)...")
    defense_results = {}

    for part_idx, params in enumerate(CONFIG["attack_parts"]):
        part_label = f"Part {part_idx + 1}"
        X_jsma_test = all_attack_results[part_label]["X_jsma"]

        # ── Experiment 3: PGD single defense (training) ─────────
        print(f"\n  [{part_label}] PGD Adversarial Training...")
        pgd_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

        # Scale PIOA optimization to match the attack perturbation magnitude
        pgd_epsilon = pioa_optimize(
            base_epsilon=params["epsilon"],
            pioa_cfg=CONFIG["pioa"]
        )
        print(f"    PIOA-optimized epsilon: {pgd_epsilon:.4f}")

        X_train_adv, y_train_adv = pgd_adversarial_training(
            pgd_model, X_train, y_train,
            epsilon=pgd_epsilon,
            alpha=params["alpha"],
            n_iter=CONFIG["pgd"]["n_iter"]
        )

        pgd_model.fit(
            X_train_adv, y_train_adv,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            class_weight=class_weights,
            verbose=1
        )

        pgd_res = evaluate_model(pgd_model, X_jsma_test, y_test_sub, label=f"PGD Defense {part_label}", X_clean=X_test_sub)
        print_results(pgd_res)

        # ── Experiment 4: SS single defense (testing) ───────────
        print(f"  [{part_label}] Spatial Smoothing Defense...")
        X_test_smoothed = spatial_smoothing(X_jsma_test, window_radius=params["sigma"])
        ss_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)
        ss_model.fit(
            X_train, y_train,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            class_weight=class_weights,
            verbose=1
        )
        ss_res = evaluate_model(ss_model, X_test_smoothed, y_test_sub, label=f"SS Defense {part_label}", X_clean=X_test_sub)
        print_results(ss_res)

        # ── Experiment 5: Hybrid defense (PGD+PIOA train + SS test + Consistency) ─
        print(f"  [{part_label}] Hybrid Defense (PGD+PIOA+SS)...")
        hybrid_model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)

        hybrid_model.fit(
            X_train_adv, y_train_adv,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["val_split"],
            class_weight=class_weights,
            verbose=1
        )

        X_hybrid_test = spatial_smoothing(X_jsma_test, window_radius=params["sigma"])

        print(f"  [{part_label}] Applying AICC + TCC Consistency Detection...")
        adv_flags = detect_adversarial(
            hybrid_model,
            X_hybrid_test,
            window_size=3,
            aicc_thresh=0.70,
            final_thresh=0.70
        )
        from evaluation import evaluate_detection

        y_adv_true = np.ones(len(adv_flags))
        det_res = evaluate_detection(adv_flags, y_adv_true)

        print("\n    -- Detection Metrics --")
        print(f"    Detection Rate   : {det_res['detection_rate']}%")
        print(f"    False Alarm Rate : {det_res['false_alarm_rate']}%")

        hybrid_res = evaluate_hybrid_defense(
            hybrid_model,
            X_hybrid_test,
            y_test_sub,
            adv_flags,
            label=f"Hybrid + AICC+TCC {part_label}",
            X_clean=X_test_sub
        )
        print_results(hybrid_res)

        defense_results[part_label] = {
            "pgd": pgd_res,
            "ss":  ss_res,
            "hybrid": hybrid_res,
        }

    # ── 5. PLOT & SAVE ──────────────────────────────────────────
    print(f"\n[5/5] Saving evaluation results and plots to {out_dir}/")
    plot_results(
        pre_attack_results,
        all_attack_results,
        defense_results,
        history,
        out_dir=str(out_dir)
    )
    print(f"      Done. Results successfully saved to: {out_dir}/")
    return defense_results


def main():
    Path(CONFIG["output_dir"]).mkdir(exist_ok=True)

    for name, file_paths in CONFIG["datasets"].items():
        resolved_files = [resolve_path(f) for f in file_paths]
        missing = [f for f in resolved_files if not Path(f).exists()]
        if missing:
            print(f"\n[SKIP] {name}: the following files were not found:")
            for m in missing:
                print(f"       {m}")
            continue
        run_pipeline(name, resolved_files)

    print("\n\nAll scenarios complete on CIC-DDoS2019 dataset!\n")


if __name__ == "__main__":
    main()