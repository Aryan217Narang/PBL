"""
run_pipeline.py
End-to-end unified pipeline replicating Barik & Misra (2025) on CIC-DDoS2019.

Usage:
    python run_pipeline.py
    python run_pipeline.py --temporal --attack_subset 1500

Steps:
  0. Preprocess / Load preprocessed dataset
  1. Train clean CNN                                    -> Scenario 1 (Baseline)
  2. Generate FGSM / JSMA / C&W adversarial test sets    -> Scenario 2 (No Defense)
  3. Train PGD-hardened CNN                              -> Scenario 3 (PGD-only)
  4. Apply Spatial Smoothing to clean CNN at test time   -> Scenario 4 (SS-only)
  5. PIOA-tune PGD hyperparams, train Hybrid CNN,        -> Scenario 5 (Hybrid & Ablations)
     and test Hybrid alone vs. Hybrid + AICC/TCC
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from data.preprocess import load_and_preprocess
from models.cnn import build_model, train_model, evaluate_accuracy
from attacks.generate_attacks import wrap_art_classifier, generate_all_attacks
from defenses.pgd_training import pgd_adversarial_train
from defenses.pioa import PIOA, build_pgd_fitness_fn
from eval.evaluate import evaluate_scenario, compute_asr, print_results_table, _predict


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f"  UNIFIED PYTORCH PIPELINE: CIC-DDoS2019 ADVERSARIAL DEFENSE")
    print(f"  Device: {device} | Temporal Stream: {args.temporal} | Smoothing Mode: {args.smoothing_mode}")
    print(f"{'='*75}\n")

    # ---------- 0. Data Loading ----------
    cache_path = os.path.join(CURRENT_DIR, "data", "processed", "processed_data.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached preprocessed data from {cache_path}...")
        data = np.load(cache_path)
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    else:
        print("Running preprocessing from raw CSVs...")
        X_train, X_test, y_train, y_test, pipeline = load_and_preprocess(
            csv_paths=args.data_glob,
            label_col=args.label_col,
            n_ica_components=args.n_ica,
            n_rfe_features=args.n_rfe,
            save_dir=os.path.join(CURRENT_DIR, "data", "processed"),
        )
    n_features = X_train.shape[1]
    print(f"Dataset shapes: train={X_train.shape}, test={X_test.shape}")

    # If --temporal mode requested, load the chronological continuous stream
    if args.temporal:
        temporal_path = os.path.join(CURRENT_DIR, "data", "processed", "temporal_test_stream.npz")
        if os.path.exists(temporal_path):
            tdata = np.load(temporal_path, allow_pickle=True)
            X_eval_stream = tdata["X"]
            y_eval_stream = tdata["y"]
            print(f"Loaded temporal evaluation stream: {len(X_eval_stream)} chronological flows")
        else:
            print("Notice: temporal_test_stream.npz not found; using standard test slice.")
            X_eval_stream = X_test
            y_eval_stream = y_test
    else:
        X_eval_stream = X_test
        y_eval_stream = y_test

    # ---------- 1. Scenario 1: Clean CNN, Pre-Attack Baseline ----------
    print("\n--- [Scenario 1] Training / Loading Clean CNN Baseline ---")
    ckpt_dir = os.path.join(CURRENT_DIR, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    clean_ckpt = os.path.join(ckpt_dir, "cnn_clean.pt")

    clean_model = build_model(n_features=n_features, device=device)
    if os.path.exists(clean_ckpt) and not args.retrain:
        print(f"Loading checkpoint: {clean_ckpt}")
        clean_model.load_state_dict(torch.load(clean_ckpt, map_location=device))
    else:
        print(f"Training clean model for {args.epochs} epochs...")
        train_model(clean_model, X_train, y_train, X_val=X_test[:2000], y_val=y_test[:2000],
                    epochs=args.epochs, device=device)
        torch.save(clean_model.state_dict(), clean_ckpt)

    results = []
    r_clean = evaluate_scenario(clean_model, X_eval_stream, y_eval_stream, device=device,
                                scenario_name="1. Pre-attack baseline (clean CNN)")
    r_clean["asr"] = 0.0
    results.append(r_clean)

    # Correctness mask needed for accurate ASR computation
    clean_preds, _ = _predict(clean_model, X_eval_stream, device=device)
    clean_correct_mask = (clean_preds == y_eval_stream)
    print(f"Clean baseline accuracy on eval set: {r_clean['accuracy']*100:.2f}%")

    # ---------- 2. Scenario 2: Generate Attacks (No Defense) ----------
    print(f"\n--- [Scenario 2] Generating Attacks (Subset={args.attack_subset}) ---")
    classifier = wrap_art_classifier(clean_model, n_features=n_features, device=device)
    adv_sets = generate_all_attacks(
        classifier,
        X_eval_stream,
        y=y_eval_stream,
        subset_size=args.attack_subset,
        sequential=True,
    )
    y_test_subset = adv_sets["y"]
    mask_subset = clean_correct_mask[:len(y_test_subset)]

    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(clean_model, adv_sets[atk_name], y_test_subset, device=device,
                              y_clean_true_for_asr=y_test_subset,
                              scenario_name=f"2. No defense vs {atk_name.upper()}")
        r["asr"] = compute_asr(clean_model, adv_sets[atk_name], y_test_subset, mask_subset, device=device)
        results.append(r)

    # ---------- 3. Scenario 3: PGD-Only Defense ----------
    print("\n--- [Scenario 3] PGD Adversarial Training ---")
    pgd_ckpt = os.path.join(ckpt_dir, "cnn_pgd.pt")
    pgd_model = build_model(n_features=n_features, device=device)
    if os.path.exists(pgd_ckpt) and not args.retrain:
        print(f"Loading checkpoint: {pgd_ckpt}")
        pgd_model.load_state_dict(torch.load(pgd_ckpt, map_location=device))
    else:
        print(f"Adversarially training PGD model for {args.epochs} epochs...")
        pgd_model.load_state_dict(clean_model.state_dict())
        pgd_adversarial_train(pgd_model, X_train, y_train, X_val=X_test[:2000], y_val=y_test[:2000],
                              eps=0.10, alpha=0.02, n_iter=7, epochs=args.epochs, device=device)
        torch.save(pgd_model.state_dict(), pgd_ckpt)

    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(pgd_model, adv_sets[atk_name], y_test_subset, device=device,
                              y_clean_true_for_asr=y_test_subset,
                              scenario_name=f"3. PGD-only defense vs {atk_name.upper()}")
        r["asr"] = compute_asr(pgd_model, adv_sets[atk_name], y_test_subset, mask_subset, device=device)
        results.append(r)

    # ---------- 4. Scenario 4: Spatial/Temporal Smoothing Only (Test-Time Filter) ----------
    print(f"\n--- [Scenario 4] Smoothing Only (Mode={args.smoothing_mode}) ---")
    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(clean_model, adv_sets[atk_name], y_test_subset, device=device,
                              use_spatial_smoothing=True, smoothing_window=args.ss_window,
                              smoothing_mode=args.smoothing_mode,
                              y_clean_true_for_asr=y_test_subset,
                              scenario_name=f"4. SS-only defense vs {atk_name.upper()}")
        r["asr"] = compute_asr(clean_model, adv_sets[atk_name], y_test_subset, mask_subset,
                               device=device, use_spatial_smoothing=True, smoothing_window=args.ss_window,
                               smoothing_mode=args.smoothing_mode)
        results.append(r)

    # ---------- 5. Scenario 5: Proposed Hybrid & Modular Consistency Ablations ----------
    print(f"\n--- [Scenario 5] Hybrid Defense (PGD+PIOA Train + {args.smoothing_mode.capitalize()} Smoothing) ---")
    hybrid_ckpt = os.path.join(ckpt_dir, "cnn_hybrid.pt")
    hybrid_model = build_model(n_features=n_features, device=device)

    if os.path.exists(hybrid_ckpt) and not args.retrain:
        print(f"Loading checkpoint: {hybrid_ckpt}")
        hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    else:
        print("Running PIOA hyperparameter optimization for PGD training...")
        bounds = {"eps": (0.02, 0.25), "alpha": (0.005, 0.04), "n_iter": (4, 12)}
        n_val = min(300, len(adv_sets["fgsm"]))
        X_val_adv = adv_sets["fgsm"][:n_val]
        y_val_adv = y_test_subset[:n_val]

        fitness_fn = build_pgd_fitness_fn(
            build_model_fn=lambda: build_model(n_features=n_features),
            X_train=X_train[:10000], y_train=y_train[:10000],
            X_val_adv=X_val_adv, y_val=y_val_adv,
            device=device, quick_epochs=args.pioa_quick_epochs,
        )
        pioa = PIOA(bounds=bounds, n_pigeons=args.pioa_pigeons,
                    n_map_compass_iters=args.pioa_iters, n_landmark_iters=args.pioa_iters)
        best_params, best_fit = pioa.optimize(fitness_fn)
        print(f"PIOA selected parameters: {best_params} (fitness={best_fit:.4f})")

        hybrid_model.load_state_dict(clean_model.state_dict())
        pgd_adversarial_train(hybrid_model, X_train, y_train, X_val=X_test[:2000], y_val=y_test[:2000],
                              eps=best_params["eps"], alpha=best_params["alpha"],
                              n_iter=best_params["n_iter"], epochs=args.epochs, device=device)
        torch.save(hybrid_model.state_dict(), hybrid_ckpt)

    for atk_name in ["fgsm", "jsma", "cw"]:
        # 5a. Hybrid alone (Primary Benchmark, no consistency)
        r_hyb = evaluate_scenario(
            hybrid_model, adv_sets[atk_name], y_test_subset, device=device,
            use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=False,
            y_clean_true_for_asr=y_test_subset,
            scenario_name=f"5a. Hybrid (Proposed) vs {atk_name.upper()}"
        )
        r_hyb["asr"] = compute_asr(
            hybrid_model, adv_sets[atk_name], y_test_subset, mask_subset,
            device=device, use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=False
        )
        results.append(r_hyb)

        # 5b. Hybrid + AICC
        r_aicc = evaluate_scenario(
            hybrid_model, adv_sets[atk_name], y_test_subset, device=device,
            use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="aicc",
            y_clean_true_for_asr=y_test_subset,
            scenario_name=f"5b. Hybrid + AICC vs {atk_name.upper()}"
        )
        r_aicc["asr"] = compute_asr(
            hybrid_model, adv_sets[atk_name], y_test_subset, mask_subset,
            device=device, use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="aicc"
        )
        results.append(r_aicc)

        # 5c. Hybrid + TCC
        r_tcc = evaluate_scenario(
            hybrid_model, adv_sets[atk_name], y_test_subset, device=device,
            use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="tcc",
            y_clean_true_for_asr=y_test_subset,
            scenario_name=f"5c. Hybrid + TCC vs {atk_name.upper()}"
        )
        r_tcc["asr"] = compute_asr(
            hybrid_model, adv_sets[atk_name], y_test_subset, mask_subset,
            device=device, use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="tcc"
        )
        results.append(r_tcc)

        # 5d. Hybrid + AICC & TCC (Combined)
        r_both = evaluate_scenario(
            hybrid_model, adv_sets[atk_name], y_test_subset, device=device,
            use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="both",
            y_clean_true_for_asr=y_test_subset,
            scenario_name=f"5d. Hybrid + AICC & TCC vs {atk_name.upper()}"
        )
        r_both["asr"] = compute_asr(
            hybrid_model, adv_sets[atk_name], y_test_subset, mask_subset,
            device=device, use_spatial_smoothing=True, smoothing_window=args.ss_window,
            smoothing_mode=args.smoothing_mode,
            use_consistency=True, consistency_mode="both"
        )
        results.append(r_both)

    # ---------- Output & Reporting ----------
    print("\n" + "="*85)
    print("                           UNIFIED BENCHMARK RESULTS")
    print("="*85)
    print_results_table(results)

    eval_dir = os.path.join(CURRENT_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    json_path = os.path.join(eval_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nDetailed JSON saved to: {json_path}")

    # Save summary CSV
    df_rows = []
    for r in results:
        df_rows.append({
            "Scenario": r["scenario"],
            "Accuracy (%)": round(r["accuracy"] * 100, 2),
            "Precision (%)": round(r["precision"] * 100, 2),
            "Recall (%)": round(r["recall"] * 100, 2),
            "Specificity (%)": round(r["specificity"] * 100, 2),
            "AUC": round(r["auc"], 4) if not np.isnan(r["auc"]) else "N/A",
            "ASR (%)": round(r["asr"] * 100, 2) if r.get("asr") is not None else "N/A",
            "False Alarm Rate (%)": round(r["false_alarm_rate"] * 100, 2) if r.get("false_alarm_rate") is not None else "N/A",
        })
    df_summary = pd.DataFrame(df_rows)
    csv_path = os.path.join(eval_dir, "summary_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PyTorch NIDS Adversarial Defense Pipeline")
    parser.add_argument("--data_glob", type=str, default="data/raw/*.csv")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--n_ica", type=int, default=30)
    parser.add_argument("--n_rfe", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--attack_subset", type=int, default=1500,
                        help="Number of samples to evaluate for attack testing")
    parser.add_argument("--ss_window", type=int, default=3)
    parser.add_argument("--smoothing_mode", type=str, default="temporal",
                        choices=["temporal", "gated_temporal", "feature"],
                        help="Smoothing defense mode: 'temporal' (axis 0), 'gated_temporal', or 'feature' (legacy)")
    parser.add_argument("--pioa_pigeons", type=int, default=10)
    parser.add_argument("--pioa_iters", type=int, default=5)
    parser.add_argument("--pioa_quick_epochs", type=int, default=2)
    parser.add_argument("--temporal", action="store_true",
                        help="Evaluate on continuous chronological temporal stream")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining of model checkpoints")
    args = parser.parse_args()
    main(args)
