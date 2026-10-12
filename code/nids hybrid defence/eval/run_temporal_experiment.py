"""
run_temporal_experiment.py
Evaluates NIDS defenses on chronological temporal stream data:
Compares:
  - Hybrid alone (PGD+PIOA + Spatial Smoothing)
  - Hybrid + AICC (Adaptive Input Consistency Check)
  - Hybrid + TCC (Temporal Consistency Check)
  - Hybrid + AICC & TCC (Combined)
Across:
  - Clean baseline
  - FGSM attack
  - JSMA attack
  - C&W attack
"""

import os
import sys
import json
import time
import numpy as np
import torch
import pandas as pd

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from models.cnn import build_model
from defenses.spatial_smoothing import spatial_smoothing
from defenses.consistency import detect_adversarial_pytorch
from attacks.generate_attacks import wrap_art_classifier, generate_all_attacks
from eval.evaluate import evaluate_scenario, compute_asr


def run_temporal_benchmark(
    temporal_stream_path: str = None,
    attack_subset: int = 1500,
    device: str = None,
    output_dir: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TEMPORAL STREAM DEFENSE EVALUATION (Device: {device}) ===")

    if temporal_stream_path is None:
        temporal_stream_path = os.path.join(PARENT_DIR, "data", "processed", "temporal_test_stream.npz")
    if output_dir is None:
        output_dir = os.path.join(PARENT_DIR, "eval")

    data = np.load(temporal_stream_path, allow_pickle=True)
    X_temporal = data["X"][:attack_subset]
    y_temporal = data["y"][:attack_subset]
    timestamps = data["timestamps"][:attack_subset] if "timestamps" in data else None

    n_features = X_temporal.shape[1]
    n_total = len(X_temporal)
    n_benign = int(np.sum(y_temporal == 0))
    n_attack = int(np.sum(y_temporal == 1))

    print(f"Loaded temporal stream: {n_total} flows")
    print(f"  Class distribution: Benign={n_benign} ({n_benign/n_total*100:.1f}%), Attack={n_attack} ({n_attack/n_total*100:.1f}%)")
    if timestamps is not None and len(timestamps) > 0:
        print(f"  Timeline: {timestamps[0]} -> {timestamps[-1]}")

    # Load trained models
    ckpt_dir = os.path.join(PARENT_DIR, "models", "checkpoints")
    clean_model = build_model(n_features=n_features, device=device)
    clean_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "cnn_clean.pt"), map_location=device))
    clean_model.eval()

    pgd_model = build_model(n_features=n_features, device=device)
    pgd_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "cnn_pgd.pt"), map_location=device))
    pgd_model.eval()

    hybrid_model = build_model(n_features=n_features, device=device)
    hybrid_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "cnn_hybrid.pt"), map_location=device))
    hybrid_model.eval()
    print("Loaded all three model checkpoints: clean, PGD, Hybrid.")

    # 1. Clean baseline evaluation
    print("\n--- Generating Clean Baseline Predictions ---")
    with torch.no_grad():
        clean_preds = clean_model(torch.tensor(X_temporal).to(device)).argmax(dim=1).cpu().numpy()
    clean_correct_mask = (clean_preds == y_temporal)
    print(f"Clean model accuracy: {np.mean(clean_correct_mask)*100:.2f}% ({np.sum(clean_correct_mask)}/{n_total} correct)")

    # 2. Generate attacks against the temporal stream
    print("\n--- Generating Adversarial Attacks (FGSM, JSMA, C&W) ---")
    classifier = wrap_art_classifier(clean_model, n_features=n_features, device=device)
    adv_dict = generate_all_attacks(
        classifier,
        X_temporal,
        y=y_temporal,
        subset_size=n_total,
        sequential=True,  # Preserves chronological flow ordering
    )

    all_scenarios = []

    # Helper evaluation function
    def eval_defense_config(model, X_input, attack_name, defense_label, use_ss=False, use_cons=False, cons_mode="both"):
        res = evaluate_scenario(
            model=model,
            X=X_input,
            y=y_temporal,
            device=device,
            use_spatial_smoothing=use_ss,
            smoothing_window=3,
            use_consistency=use_cons,
            consistency_mode=cons_mode,
            y_clean_true_for_asr=y_temporal if attack_name != "Clean" else None,
            scenario_name=f"{defense_label} vs {attack_name}",
        )
        if attack_name != "Clean":
            res["asr"] = compute_asr(
                model=model,
                X_adv=X_input,
                y_true=y_temporal,
                X_clean_correct_mask=clean_correct_mask,
                device=device,
                use_spatial_smoothing=use_ss,
                smoothing_window=3,
                use_consistency=use_cons,
                consistency_mode=cons_mode,
            )
        else:
            res["asr"] = 0.0

        res["attack"] = attack_name
        res["defense"] = defense_label
        return res

    attacks = [
        ("Clean", X_temporal),
        ("FGSM", adv_dict["fgsm"]),
        ("JSMA", adv_dict["jsma"]),
        ("CW", adv_dict["cw"]),
    ]

    for atk_name, X_atk in attacks:
        print(f"\n==================== Evaluating: {atk_name} ====================")
        
        # 1. Undefended Baseline
        r_undef = eval_defense_config(clean_model, X_atk, atk_name, "1. No Defense", use_ss=False, use_cons=False)
        all_scenarios.append(r_undef)

        # 2. SS-only
        r_ss = eval_defense_config(clean_model, X_atk, atk_name, "2. SS-only", use_ss=True, use_cons=False)
        all_scenarios.append(r_ss)

        # 3. PGD-only
        r_pgd = eval_defense_config(pgd_model, X_atk, atk_name, "3. PGD-only", use_ss=False, use_cons=False)
        all_scenarios.append(r_pgd)

        # 4. Hybrid Defense alone (Primary Benchmark: PGD+PIOA trained + Spatial Smoothing)
        r_hybrid = eval_defense_config(hybrid_model, X_atk, atk_name, "4. Hybrid (Proposed)", use_ss=True, use_cons=False)
        all_scenarios.append(r_hybrid)

        # 5. Hybrid + AICC (Input Consistency only)
        r_hyb_aicc = eval_defense_config(hybrid_model, X_atk, atk_name, "5. Hybrid + AICC", use_ss=True, use_cons=True, cons_mode="aicc")
        all_scenarios.append(r_hyb_aicc)

        # 6. Hybrid + TCC (Temporal Consistency only)
        r_hyb_tcc = eval_defense_config(hybrid_model, X_atk, atk_name, "6. Hybrid + TCC", use_ss=True, use_cons=True, cons_mode="tcc")
        all_scenarios.append(r_hyb_tcc)

        # 7. Hybrid + AICC & TCC (Combined)
        r_hyb_both = eval_defense_config(hybrid_model, X_atk, atk_name, "7. Hybrid + AICC & TCC", use_ss=True, use_cons=True, cons_mode="both")
        all_scenarios.append(r_hyb_both)

        print(f"  No Defense            : Acc={r_undef['accuracy']*100:.2f}%, ASR={r_undef['asr']*100:.2f}%")
        print(f"  Hybrid (Proposed)     : Acc={r_hybrid['accuracy']*100:.2f}%, ASR={r_hybrid['asr']*100:.2f}%")
        print(f"  Hybrid + AICC         : Acc={r_hyb_aicc['accuracy']*100:.2f}%, ASR={r_hyb_aicc['asr']*100:.2f}%")
        print(f"  Hybrid + TCC          : Acc={r_hyb_tcc['accuracy']*100:.2f}%, ASR={r_hyb_tcc['asr']*100:.2f}%")
        print(f"  Hybrid + AICC & TCC   : Acc={r_hyb_both['accuracy']*100:.2f}%, ASR={r_hyb_both['asr']*100:.2f}%")

    # Save to JSON
    json_path = os.path.join(output_dir, "temporal_experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(all_scenarios, f, indent=2, default=float)
    print(f"\nDetailed results saved to {json_path}")

    # Build Summary DataFrame
    df_rows = []
    for r in all_scenarios:
        df_rows.append({
            "Attack": r["attack"],
            "Defense": r["defense"],
            "Accuracy (%)": round(r["accuracy"] * 100, 2),
            "Precision (%)": round(r["precision"] * 100, 2),
            "Recall (%)": round(r["recall"] * 100, 2),
            "Specificity (%)": round(r["specificity"] * 100, 2),
            "AUC": round(r["auc"], 4) if not np.isnan(r["auc"]) else "N/A",
            "ASR (%)": round(r["asr"] * 100, 2),
            "Detection Rate (%)": round(r["detection_rate"] * 100, 2) if r["detection_rate"] is not None else "N/A",
            "False Alarm Rate (%)": round(r["false_alarm_rate"] * 100, 2) if r["false_alarm_rate"] is not None else "N/A",
        })
    df_results = pd.DataFrame(df_rows)
    csv_path = os.path.join(output_dir, "temporal_experiment_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Print clean side-by-side comparison tables
    print("\n" + "=" * 90)
    print("                    SIDE-BY-SIDE TEMPORAL DEFENSE COMPARISON")
    print("=" * 90)

    for atk in ["Clean", "FGSM", "JSMA", "CW"]:
        sub = df_results[df_results["Attack"] == atk]
        print(f"\n>>> Traffic Condition: {atk} Attack Stream")
        print(f"{'Defense Configuration':<32}{'Acc (%)':>9}{'Prec (%)':>10}{'Rec (%)':>9}{'Spec (%)':>10}{'ASR (%)':>9}{'FAR (%)':>9}")
        print("-" * 88)
        for _, row in sub.iterrows():
            far_str = str(row["False Alarm Rate (%)"])
            print(f"{row['Defense']:<32}{row['Accuracy (%)']:>9.2f}{row['Precision (%)']:>10.2f}"
                  f"{row['Recall (%)']:>9.2f}{row['Specificity (%)']:>10.2f}{row['ASR (%)']:>9.2f}{far_str:>9}")

    return df_results


if __name__ == "__main__":
    run_temporal_benchmark()
