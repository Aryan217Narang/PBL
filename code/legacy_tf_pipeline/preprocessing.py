"""
preprocessing.py
Steps (from paper Fig. 1):
  Data Loading -> Data Cleaning -> Feature Normalization (MinMaxScaler) ->
  Binary Intrusion Label Encoding (BENIGN=0, ATTACK=1) ->
  Feature Extraction (ICA, 30 components) -> Feature Selection (RFE, 25 features) ->
  Train/Test Split (80/20 Stratified)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABEL_COLUMNS = [
    "Label", "label", " Label",
    "attack_cat", "class",
]

DROP_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "SimillarHTTP", "Unnamed: 0",
    "Inbound", "Source Port", "Destination Port"
]

ROWS_PER_FILE = 15000
CACHE_FILE = "data/processed_cicddos2019.npz"


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    thresh = int(0.5 * len(df))
    df = df.dropna(axis=1, thresh=thresh)
    df = df.fillna(df.median(numeric_only=True))
    return df


def _drop_non_numeric(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols + [label_col]]


def load_and_preprocess(
    file_paths: list,
    test_size: float = 0.20,
    n_ica_components: int = 30,
    n_rfe_features: int = 25,
    random_state: int = 42,
    rows_per_file: int = ROWS_PER_FILE,
    force_recompute: bool = False,
):
    # Check if cached preprocessed dataset already exists
    if not force_recompute and os.path.exists(CACHE_FILE):
        print(f"    Loading cached preprocessed data from {CACHE_FILE}...")
        try:
            cached = np.load(CACHE_FILE, allow_pickle=True)
            X_train = cached["X_train"]
            X_test  = cached["X_test"]
            y_train = cached["y_train"]
            y_test  = cached["y_test"]
            n_classes = int(cached["n_classes"])
            class_weight_dict = {
                int(k): float(v) for k, v in cached["class_weights"].item().items()
            } if "class_weights" in cached else None
            print(f"    Loaded from cache: Train={X_train.shape}, Test={X_test.shape}, Classes={n_classes}")
            return X_train, X_test, y_train, y_test, n_classes, class_weight_dict
        except Exception as e:
            print(f"    Cache read failed ({e}), recomputing from CSVs...")

    # ── Load each file with nrows — fast & memory-safe ────────
    dfs = []
    for f in file_paths:
        print(f"    Loading: {f}")
        try:
            df = pd.read_csv(
                f,
                nrows=rows_per_file,
                low_memory=False,
                on_bad_lines='skip'
            )
            df = _clean_column_names(df)
            print(f"      Loaded {len(df):,} rows, {df.shape[1]} columns")
            dfs.append(df)
        except Exception as e:
            print(f"      ERROR: {e} - skipping")
            continue

    if not dfs:
        raise RuntimeError("No files were loaded successfully.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"    Total combined shape: {df.shape}")

    # ── Drop non-generalizable metadata columns ─────────────────
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    # ── Detect label column ──────────────────────────────────────
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    print(f"    Label column: '{label_col}'")

    # ── Clean & Filter Numerics ──────────────────────────────────
    df = _clean(df)
    df = _drop_non_numeric(df, label_col)

    # ── Separate features and labels ─────────────────────────────
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    y_raw = df[label_col].values

    # ── Binary Intrusion Detection Label Encoding ────────────────
    # 0 = BENIGN (Normal), 1 = ATTACK (DDoS Intrusion)
    y = np.array([0 if str(v).strip().upper() == "BENIGN" else 1 for v in y_raw], dtype=np.int32)
    n_classes = 2
    benign_cnt = int(np.sum(y == 0))
    attack_cnt = int(np.sum(y == 1))
    print(f"    Binary Classes (2): BENIGN={benign_cnt:,} ({benign_cnt/len(y)*100:.1f}%), ATTACK={attack_cnt:,} ({attack_cnt/len(y)*100:.1f}%)")

    # ── MinMax Normalization ─────────────────────────────────────
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # ── ICA: Feature Extraction (30 components) ──────────────────
    n_ica = min(n_ica_components, X.shape[1], X.shape[0] - 1)
    ica = FastICA(n_components=n_ica, random_state=random_state, max_iter=500, tol=0.01)
    X_ica = ica.fit_transform(X)
    print(f"    ICA extracted components: {X_ica.shape[1]}")

    # ── RFE: Feature Selection (25 features) ─────────────────────
    n_rfe = min(n_rfe_features, X_ica.shape[1])
    estimator = RandomForestClassifier(n_estimators=20, random_state=random_state, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_rfe, step=1)
    X_rfe = rfe.fit_transform(X_ica, y)
    print(f"    RFE selected top discriminative features: {X_rfe.shape[1]}")

    # ── Train / Test Split (80/20 Stratified) ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_rfe, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Compute Balanced Class Weights ───────────────────────────
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"    Balanced Class Weights: Benign={cw[0]:.3f}, Attack={cw[1]:.3f}")

    # ── Reshape for 1D-CNN (samples, features, 1) ────────────────
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    # ── Cache to disk for instant future runs ────────────────────
    os.makedirs("data", exist_ok=True)
    np.savez(
        CACHE_FILE,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=n_classes,
        class_weights=class_weight_dict,
    )
    print(f"    Cached preprocessed dataset to: {CACHE_FILE}")

    return X_train, X_test, y_train, y_test, n_classes, class_weight_dict