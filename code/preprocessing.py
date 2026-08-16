"""
preprocessing.py
Steps (from paper Fig. 1):
  Data Loading → Data Cleaning → Feature Normalization →
  Label Encoding → Feature Selection (RFE) → Feature Extraction (ICA) →
  Train/Test Split
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import FastICA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ── Known label columns ──────────────────────────────────────────────────────
LABEL_COLUMNS = [
    "Label", "label", " Label",
    "attack_cat", "class",
]

DROP_COLUMNS = [
    "Flow ID", " Flow ID", "Source IP", " Source IP", "Destination IP",
    " Destination IP", "Timestamp", " Timestamp", "SimillarHTTP",
]

# How many rows to read from each CSV — fast, no pre-scanning
ROWS_PER_FILE = 20_000


def _detect_label_column(df: pd.DataFrame) -> str:
    for candidate in LABEL_COLUMNS:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


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
    n_ica_components: int = 20,
    n_rfe_features: int = 15,
    random_state: int = 42,
    rows_per_file: int = ROWS_PER_FILE,
):
    # ── Load each file with nrows — fast, no pre-scanning ────────
    dfs = []
    for f in file_paths:
        print(f"    Loading: {f}")
        try:
            df = pd.read_csv(
                f,
                nrows=rows_per_file,        # just read the first N rows, instantly
                low_memory=False,
                on_bad_lines='skip'
            )
            print(f"      Loaded {len(df):,} rows, {df.shape[1]} columns")
            dfs.append(df)
        except Exception as e:
            print(f"      ERROR: {e} — skipping")
            continue

    if not dfs:
        raise RuntimeError("No files were loaded successfully.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"    Total combined shape: {df.shape}")

    # ── Drop metadata columns ────────────────────────────────────
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    # ── Detect label column ──────────────────────────────────────
    label_col = _detect_label_column(df)
    print(f"    Label column: '{label_col}'")

    # ── Clean ────────────────────────────────────────────────────
    df = _clean(df)
    df = _drop_non_numeric(df, label_col)

    # ── Separate features and labels ─────────────────────────────
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    y_raw = df[label_col].values

    # ── Label Encoding ───────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    n_classes = len(le.classes_)
    print(f"    Classes ({n_classes}): {list(le.classes_[:5])}{'...' if n_classes > 5 else ''}")

    # ── MinMax Normalization ─────────────────────────────────────
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # ── ICA: Feature Extraction ──────────────────────────────────
    n_ica = min(n_ica_components, X.shape[1], X.shape[0] - 1)
    ica = FastICA(n_components=n_ica, random_state=random_state, max_iter=500, tol=0.01)
    X_ica = ica.fit_transform(X)
    print(f"    ICA components: {X_ica.shape[1]}")

    # ── RFE: Feature Selection ───────────────────────────────────
    n_rfe = min(n_rfe_features, X_ica.shape[1])
    estimator = RandomForestClassifier(n_estimators=20, random_state=random_state, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_rfe, step=1)
    X_rfe = rfe.fit_transform(X_ica, y)
    print(f"    RFE selected features: {X_rfe.shape[1]}")

    # ── Train / Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_rfe, y, test_size=test_size, random_state=random_state, 
    )

    # ── Reshape for CNN (samples, features, 1) ───────────────────
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, X_test, y_train, y_test, n_classes