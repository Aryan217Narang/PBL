"""
End-to-end pipeline replicating Barik & Misra (2025) on CIC-DDoS2019.

    python run_pipeline.py --data_glob "data/raw/*.csv"

Steps:
  0. Preprocess (MinMaxScaler -> ICA -> RFE)
  1. Train clean CNN                                    -> Scenario 1
  2. Generate FGSM / JSMA / C&W adversarial test sets    -> Scenario 2
  3. Train PGD-hardened CNN                              -> Scenario 3
  4. Apply Spatial Smoothing to clean CNN at test time   -> Scenario 4
  5. PIOA-tune PGD hyperparams, retrain, + SS at test    -> Scenario 5 (proposed)
"""

import argparse
import os
import numpy as np
import torch

from data.preprocess import load_and_preprocess
from models.cnn import build_model, train_model, evaluate_accuracy
from attacks.generate_attacks import wrap_art_classifier, generate_all_attacks
from defenses.pgd_training import pgd_adversarial_train
from defenses.pioa import PIOA, build_pgd_fitness_fn
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from eval.evaluate import evaluate_scenario, compute_asr, print_results_table


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------- 0. Preprocess ----------
    cache_path = "data/processed/processed_data.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached preprocessed data from {cache_path}...")
        data = np.load(cache_path)
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    else:
        X_train, X_test, y_train, y_test, pipeline = load_and_preprocess(
            csv_paths=args.data_glob,
            label_col=args.label_col,
            n_ica_components=args.n_ica,
            n_rfe_features=args.n_rfe,
            save_dir="data/processed",
        )
    n_features = X_train.shape[1]
    print(f"Preprocessed shapes: train={X_train.shape} test={X_test.shape}")

    # ---------- 1. Scenario 1: clean CNN, pre-attack baseline ----------
    clean_model = build_model(n_features=n_features, device=device)
    train_model(clean_model, X_train, y_train, X_val=X_test, y_val=y_test,
                epochs=args.epochs, device=device)
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(clean_model.state_dict(), "models/checkpoints/cnn_clean.pt")

    results = []
    results.append(evaluate_scenario(clean_model, X_test, y_test, device=device,
                                      scenario_name="1. Pre-attack baseline (clean CNN)"))

    # correctness mask needed for proper ASR later
    from eval.evaluate import _predict
    clean_preds, _ = _predict(clean_model, X_test, device=device)
    clean_correct_mask = clean_preds == y_test

    # ---------- 2. Scenario 2: attacks, no defense ----------
    classifier = wrap_art_classifier(clean_model, n_features=n_features, device=device)
    adv_sets = generate_all_attacks(classifier, X_test, subset_size=args.attack_subset)
    y_test_subset = y_test[:len(adv_sets["clean"])] if args.attack_subset else y_test
    mask_subset = clean_correct_mask[:len(adv_sets["clean"])] if args.attack_subset else clean_correct_mask

    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(clean_model, adv_sets[atk_name], y_test_subset, device=device,
                               scenario_name=f"2. No defense vs {atk_name.upper()}")
        r["asr"] = compute_asr(clean_model, adv_sets[atk_name], y_test_subset, mask_subset, device=device)
        results.append(r)

    # ---------- 3. Scenario 3: PGD-only defense ----------
    pgd_model = build_model(n_features=n_features, device=device)
    pgd_model.load_state_dict(clean_model.state_dict())  # warm start, optional
    pgd_adversarial_train(pgd_model, X_train, y_train, X_val=X_test, y_val=y_test,
                           eps=0.1, alpha=0.02, n_iter=7, epochs=args.epochs, device=device)
    torch.save(pgd_model.state_dict(), "models/checkpoints/cnn_pgd.pt")

    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(pgd_model, adv_sets[atk_name], y_test_subset, device=device,
                               scenario_name=f"3. PGD-only defense vs {atk_name.upper()}")
        r["asr"] = compute_asr(pgd_model, adv_sets[atk_name], y_test_subset, mask_subset, device=device)
        results.append(r)

    # ---------- 4. Scenario 4: Spatial Smoothing only (clean model, test-time filter) ----------
    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(clean_model, adv_sets[atk_name], y_test_subset, device=device,
                               use_spatial_smoothing=True, smoothing_window=args.ss_window,
                               scenario_name=f"4. SS-only defense vs {atk_name.upper()}")
        results.append(r)

    # ---------- 5. Scenario 5: Proposed hybrid (PGD + PIOA train, SS test) ----------
    print("\nRunning PIOA to optimize PGD hyperparameters (eps, alpha, n_iter)...")
    bounds = {"eps": (0.02, 0.3), "alpha": (0.005, 0.05), "n_iter": (3, 15)}

    # small validation adversarial set for PIOA's fitness function (keep it cheap)
    n_val = min(300, len(adv_sets["fgsm"]))
    X_val_adv = adv_sets["fgsm"][:n_val]
    y_val_adv = y_test_subset[:n_val]

    fitness_fn = build_pgd_fitness_fn(
        build_model_fn=lambda: build_model(n_features=n_features),
        X_train=X_train, y_train=y_train,
        X_val_adv=X_val_adv,
        y_val=y_val_adv,
        device=device, quick_epochs=args.pioa_quick_epochs,
    )
    pioa = PIOA(bounds=bounds, n_pigeons=args.pioa_pigeons,
                n_map_compass_iters=args.pioa_iters, n_landmark_iters=args.pioa_iters)
    best_params, best_fit = pioa.optimize(fitness_fn)
    print(f"PIOA best params: {best_params} (fitness={best_fit:.4f})")

    hybrid_model = build_model(n_features=n_features, device=device)
    pgd_adversarial_train(hybrid_model, X_train, y_train, X_val=X_test, y_val=y_test,
                           eps=best_params["eps"], alpha=best_params["alpha"],
                           n_iter=best_params["n_iter"], epochs=args.epochs, device=device)
    torch.save(hybrid_model.state_dict(), "models/checkpoints/cnn_hybrid.pt")

    for atk_name in ["fgsm", "jsma", "cw"]:
        r = evaluate_scenario(hybrid_model, adv_sets[atk_name], y_test_subset, device=device,
                               use_spatial_smoothing=True, smoothing_window=args.ss_window,
                               scenario_name=f"5. Hybrid (PGD+PIOA+SS) vs {atk_name.upper()}")
        r["asr"] = compute_asr(hybrid_model, adv_sets[atk_name], y_test_subset, mask_subset, device=device)
        results.append(r)

    print("\n===== FINAL RESULTS =====")
    print_results_table(results)

    import json
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved detailed results to eval/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, default="data/raw/*.csv",
                         help="Glob pattern for CIC-DDoS2019 CSV files")
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--n_ica", type=int, default=30)
    parser.add_argument("--n_rfe", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--attack_subset", type=int, default=2000,
                         help="Number of test samples to attack (JSMA/C&W are slow)")
    parser.add_argument("--ss_window", type=int, default=3)
    parser.add_argument("--pioa_pigeons", type=int, default=10)
    parser.add_argument("--pioa_iters", type=int, default=5)
    parser.add_argument("--pioa_quick_epochs", type=int, default=3,
                         help="Epochs used per PIOA fitness evaluation (keep small)")
    args = parser.parse_args()
    main(args)
