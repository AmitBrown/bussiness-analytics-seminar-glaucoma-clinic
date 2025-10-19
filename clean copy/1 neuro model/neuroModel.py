import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import json
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.base import clone
import joblib

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

CONFIG = {
    "LOAD_FROM_CSV": True,
    "POSITIVE_CSV": "labeled_neuro.csv",
    "UNLABELED_CSV": "untagged.csv",
    "NEGATIVE_CSV": "glaucoma_tagged.csv",
    "ID_COLUMNS": [
        'Patient ID',
        'Study Instance UID',
        'Date of Birth',
        'OCT Scan Date'
    ],
    "TIMESTAMP_COLUMN": None,
    "USE_CONFIDENT_NEGATIVES": True,
    "TARGET_PRECISION": 0.95,
    "XGB_PARAMS": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42,
        "scale_pos_weight": 1.0,  # set per split below
    },
    "OUTPUT_SCORES_CSV": "unlabeled_scores.csv",
    "OUTPUT_FLAGGED_CSV": "unlabeled_flagged_neurolike.csv",
    "ARTIFACT_MODEL": "best_model.joblib",
    "ARTIFACT_JSON": "model_meta.json",
}

# -----------------------
# IO
# -----------------------
def load_csvs(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir())
    required_files = [cfg["POSITIVE_CSV"], cfg["UNLABELED_CSV"]]
    missing = [f for f in required_files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {missing}. Please check the filenames and directory.")
    pos = pd.read_csv(cfg["POSITIVE_CSV"])
    unl = pd.read_csv(cfg["UNLABELED_CSV"])
    neg = None
    if cfg["NEGATIVE_CSV"] and os.path.exists(cfg["NEGATIVE_CSV"]):
        neg = pd.read_csv(cfg["NEGATIVE_CSV"])
    return pos, unl, neg

# -----------------------
# Features / Preprocess
# -----------------------
def infer_feature_lists(df: pd.DataFrame, id_cols: list, labels_to_drop: list) -> Tuple[list, list]:
    drop_cols = [c for c in (id_cols + labels_to_drop) if c in df.columns]
    drop_cols += [c for c in df.columns if c.endswith('_dob') or c.endswith('_scan_dt')]
    X = df.drop(columns=drop_cols, errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols

def make_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

# -----------------------
# Splits
# -----------------------
def group_patient_split(df, y_col, patient_col, test_size=0.3, seed=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[patient_col].fillna("MISSING")
    idx = np.arange(len(df))
    for train_idx, test_idx in splitter.split(idx, df[y_col], groups):
        X = df.drop(columns=[y_col], errors="ignore")
        y = df[y_col]
        if y.iloc[test_idx].sum() < 10:
            print("WARNING: Test set has fewer than 10 positives.")
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return train_test_split(df.drop(columns=[y_col], errors="ignore"), df[y_col],
                            test_size=test_size, random_state=seed, stratify=df[y_col])

def time_or_random_split(df: pd.DataFrame, y_col: str, ts_col: Optional[str], test_size=0.3, seed=42):
    if 'Patient ID' in df.columns:
        return group_patient_split(df, y_col, 'Patient ID', test_size=test_size, seed=seed)
    X = df.drop(columns=[y_col], errors="ignore")
    y = df[y_col]
    if ts_col and ts_col in df.columns:
        df_sorted = df.sort_values(ts_col)
        cutoff = int(len(df_sorted) * (1 - test_size))
        tr_idx = df_sorted.index[:cutoff]
        te_idx = df_sorted.index[cutoff:]
        return X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# -----------------------
# PU calibration helpers
# -----------------------
def estimate_c(pipe: Pipeline, X_pos_hold: pd.DataFrame) -> float:
    p_s = pipe.predict_proba(X_pos_hold)[:, 1]
    return float(np.clip(p_s.mean(), 1e-6, 1.0))

def p_true_from_ps(p_s: np.ndarray, c: float) -> np.ndarray:
    return np.clip(p_s / max(c, 1e-6), 0.0, 1.0)

def estimate_oof_c(pipe, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        pipe_fold = clone(pipe)
        pipe_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = pipe_fold.predict_proba(X.iloc[test_idx])[:, 1]
    c = oof_preds[y == 1].mean()
    return float(np.clip(c, 1e-6, 1.0))

# -----------------------
# Thresholding / metrics
# -----------------------
def choose_tau_for_precision(y_true_pos_mask: np.ndarray, p_scores: np.ndarray, target_precision: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true_pos_mask, p_scores)
    tau = None
    for pr, th in zip(precision[:-1], thresholds):
        if pr >= target_precision:
            tau = float(th)
            break
    return 0.9 if tau is None else tau

def report_metrics(name: str, pr_auc: float, precision_at_tau: float, recall_pos: float, tau: float, extra: Dict[str, Any] = None):
    msg = f"[{name}] PR_AUC={pr_auc:.4f}  Precision@tau={precision_at_tau:.4f}  Recall_pos={recall_pos:.4f}  tau={tau:.4f}"
    if extra:
        msg += " " + " ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items()])
    print(msg)

# -----------------------
# Model trainers
# -----------------------
def train_eval_PU(base_clf, preproc, X_train, y_train, X_test, y_test, target_precision=0.95, seed=42):
    pipe = Pipeline([("prep", preproc), ("clf", base_clf)])
    pipe.fit(X_train, y_train)
    pos_mask = (y_train == 1)
    X_pos_train = X_train[pos_mask]
    X_pos_hold = X_pos_train.sample(frac=0.2, random_state=seed) if len(X_pos_train) > 50 else X_pos_train
    c = estimate_c(pipe, X_pos_hold)
    p_s_test = pipe.predict_proba(X_test)[:, 1]
    p_true_test = p_true_from_ps(p_s_test, c)
    tau = choose_tau_for_precision(y_test.values, p_true_test, target_precision)
    pr_auc = float(average_precision_score(y_test, p_true_test))
    y_pred = (p_true_test >= tau).astype(int)
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    pred_pos = int((y_pred == 1).sum())
    actual_pos = int((y_test == 1).sum())
    precision_at_tau = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall_pos = float(tp / actual_pos) if actual_pos > 0 else 0.0
    report_metrics("PU", pr_auc, precision_at_tau, recall_pos, tau, extra={"c": c})
    return {"pipeline": pipe, "pr_auc": pr_auc, "precision_at_tau": precision_at_tau, "recall_pos": recall_pos, "tau": tau, "c": c}

def train_eval_PNU(base_clf, preproc, df_pos: pd.DataFrame, df_neg: pd.DataFrame, id_cols: list,
                   ts_col: Optional[str], target_precision=0.95, seed=42):
    y_col = "y_supervised"
    df_pos_sup = df_pos.copy(); df_pos_sup[y_col] = 1
    df_neg_sup = df_neg.copy(); df_neg_sup[y_col] = 0
    df_lab = pd.concat([df_pos_sup, df_neg_sup], ignore_index=True)
    num_cols, cat_cols = infer_feature_lists(df_lab, id_cols, [y_col])
    preproc_sup = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = time_or_random_split(df_lab, y_col, ts_col, test_size=0.3, seed=seed)
    pipe = Pipeline([("prep", preproc_sup), ("clf", base_clf)])
    pipe.fit(X_train, y_train)
    p_true_test = pipe.predict_proba(X_test)[:, 1]
    tau = choose_tau_for_precision(y_test.values, p_true_test, target_precision)
    pr_auc = float(average_precision_score(y_test, p_true_test))
    y_pred = (p_true_test >= tau).astype(int)
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    pred_pos = int((y_pred == 1).sum())
    actual_pos = int((y_pred == 1).sum())
    precision_at_tau = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall_pos = float(tp / actual_pos) if actual_pos > 0 else 0.0
    report_metrics("PNU", pr_auc, precision_at_tau, recall_pos, tau)
    return {"pipeline": pipe, "pr_auc": pr_auc, "precision_at_tau": precision_at_tau, "recall_pos": recall_pos, "tau": tau, "c": 1.0, "num_cols": num_cols, "cat_cols": cat_cols}

# -----------------------
# Scoring & outputs (FIXED)
# -----------------------
def score_and_save_unlabeled(best: Dict[str, Any],
                             df_unl: pd.DataFrame,
                             id_cols: list,
                             out_scores: str,
                             out_flagged: str):
    """
    Predict on unlabeled rows and save:
      - scores with ID columns
      - full flagged rows (all original columns)
    IDs are pulled from df_unl (NOT from the feature matrix).
    """
    pipe = best["pipeline"]
    tau = best["tau"]
    c   = best.get("c", 1.0)

    # 1) Keep ID cols from the original df_unl
    keep_id_cols = [cname for cname in id_cols if cname in df_unl.columns]
    ids_df = df_unl[keep_id_cols].copy() if keep_id_cols else pd.DataFrame(index=df_unl.index)
    if not keep_id_cols:
        ids_df["__row__"] = np.arange(len(df_unl))  # fallback key

    # 2) Build feature matrix (drop IDs + any label cols)
    drop_cols = keep_id_cols + [c for c in ["is_neuro_labeled", "y_supervised"] if c in df_unl.columns]
    X_unl = df_unl.drop(columns=drop_cols, errors="ignore")

    # 3) Predict and threshold
    p_s_unl   = pipe.predict_proba(X_unl)[:, 1]
    p_true_unl = np.clip(p_s_unl / max(c, 1e-6), 0.0, 1.0)
    flags     = (p_true_unl >= tau).astype(int)

    # 4) Assemble outputs
    scores_df = ids_df.copy()
    scores_df["p_neuro"] = p_true_unl
    scores_df["flag_neuro_like"] = flags
    scores_df.to_csv(out_scores, index=False)

    # Save full original columns for flagged rows
    flagged_df = df_unl.loc[flags == 1].copy()
    flagged_df.to_csv(out_flagged, index=False)

    print(f"Saved scores: {out_scores}  |  Flagged rows: {len(flagged_df)} -> {out_flagged}")
    return scores_df, flagged_df

# -----------------------
# Missingness handling
# -----------------------
def summarize_missing(df, name, topk=15):
    miss = df.isna().mean().sort_values(ascending=False)
    print(f"\n[{name}] Missingness summary (top {topk}):")
    print((miss.head(topk) * 100).round(1).astype(str) + "%")
    print(f"Rows with any NaN: {df.isna().any(axis=1).sum()} / {len(df)}")

def add_missing_flags(df, exclude_cols=None, min_rate_to_flag=0.0):
    exclude_cols = set(exclude_cols or [])
    rates = df.isna().mean()
    cols_to_flag = [c for c, r in rates.items() if r >= min_rate_to_flag and c not in exclude_cols]
    for c in cols_to_flag:
        df[f"{c}__isna"] = df[c].isna().astype(int)
    return df

def drop_sparse_cols(df, threshold=0.90):
    rates = df.isna().mean()
    drop = rates[rates > threshold].index.tolist()
    if drop:
        print(f"Dropping very-sparse columns (> {int(threshold*100)}% NaN): {drop}")
        df = df.drop(columns=drop)
    return df

def drop_sparse_rows(df, threshold=0.60, key_cols=None):
    row_nan_rate = df.isna().mean(axis=1)
    to_drop = row_nan_rate > threshold
    if key_cols:
        for kc in key_cols:
            if kc in df.columns:
                to_drop |= df[kc].isna()
    n = int(to_drop.sum())
    if n:
        print(f"Dropping {n} super-sparse rows (> {int(threshold*100)}% NaN or missing key col).")
    return df.loc[~to_drop].copy()

def prepare_missingness(df_pos, df_neg, df_unl, id_cols, ts_col):
    summarize_missing(df_pos, "NEURO (positives)")
    if df_neg is not None:
        summarize_missing(df_neg, "Glaucoma (confident negatives)")
    summarize_missing(df_unl, "UNLABELED")

    all_cols = set(df_pos.columns) | set(df_unl.columns) | (set(df_neg.columns) if df_neg is not None else set())
    df_pos = df_pos.reindex(columns=sorted(all_cols))
    df_unl = df_unl.reindex(columns=sorted(all_cols))
    if df_neg is not None:
        df_neg = df_neg.reindex(columns=sorted(all_cols))

    exclude = set((id_cols or [])) | ({ts_col} if ts_col else set())
    df_pos = add_missing_flags(df_pos, exclude_cols=exclude, min_rate_to_flag=0.0)
    df_unl = add_missing_flags(df_unl, exclude_cols=exclude, min_rate_to_flag=0.0)
    if df_neg is not None:
        df_neg = add_missing_flags(df_neg, exclude_cols=exclude, min_rate_to_flag=0.0)

    df_pos = drop_sparse_cols(df_pos, threshold=0.90)
    df_unl = drop_sparse_cols(df_unl, threshold=0.90)
    if df_neg is not None:
        df_neg = drop_sparse_cols(df_neg, threshold=0.90)

    key_cols = (id_cols or []) + ([ts_col] if ts_col else [])
    df_pos = drop_sparse_rows(df_pos, threshold=0.60, key_cols=key_cols)
    df_unl = drop_sparse_rows(df_unl, threshold=0.60, key_cols=key_cols)
    if df_neg is not None:
        df_neg = drop_sparse_rows(df_neg, threshold=0.60, key_cols=key_cols)

    return df_pos, df_neg, df_unl

# -----------------------
# Feature engineering (robust Laterality)
# -----------------------
def engineer_date_features(df):
    df = df.copy()
    dob = pd.to_datetime(df.get('Date of Birth'), errors='coerce')
    scan = pd.to_datetime(df.get('OCT Scan Date'), errors='coerce')
    age = ((scan - dob).dt.days / 365.25).clip(0, 110)
    df['age_at_scan'] = age.astype(float)
    df['scan_year'] = scan.dt.year
    df['scan_month'] = scan.dt.month
    df['scan_dow'] = scan.dt.dayofweek

    # Normalize Laterality robustly to OD/OS if present
    if 'Laterality' in df.columns:
        lat = df['Laterality'].astype(str).str.strip().str.upper()
        mapping = {
            'OD': 'OD', 'O.D.': 'OD', 'RIGHT': 'OD', 'R': 'OD',
            'OS': 'OS', 'O.S.': 'OS', 'LEFT': 'OS', 'L': 'OS'
        }
        df['Laterality'] = lat.map(mapping)
        # If mapping fails (unrecognized), keep as NaN; missing flags will capture it

    # Drop raw dates and helpers (they are in ID list anyway)
    for col in ['Date of Birth', 'OCT Scan Date']:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in list(df.columns):
        if col.endswith('_dob') or col.endswith('_scan_dt'):
            df = df.drop(columns=[col])
    return df

# -----------------------
# Main
# -----------------------
def main(df_pos: pd.DataFrame = None, df_unl: pd.DataFrame = None,
         df_neg: Optional[pd.DataFrame] = None, cfg: Dict[str, Any] = CONFIG):

    if cfg["LOAD_FROM_CSV"]:
        df_pos, df_unl, df_neg = load_csvs(cfg)
    else:
        if df_pos is None or df_unl is None:
            raise ValueError("Provide df_pos and df_unl when LOAD_FROM_CSV=False")

    id_cols = cfg["ID_COLUMNS"]
    ts_col = cfg["TIMESTAMP_COLUMN"]
    target_precision = cfg["TARGET_PRECISION"]

    # 1) Feature engineering FIRST
    df_pos = engineer_date_features(df_pos)
    df_unl = engineer_date_features(df_unl)
    if df_neg is not None:
        df_neg = engineer_date_features(df_neg)

    # 2) Missingness SECOND
    df_pos, df_neg, df_unl = prepare_missingness(df_pos, df_neg, df_unl, id_cols, ts_col)

    # 3) Dynamic XGB balance
    if _HAS_XGB:
        n_pos_all = len(df_pos)
        n_neg_all = len(df_neg) if df_neg is not None else len(df_unl)
        CONFIG["XGB_PARAMS"]["scale_pos_weight"] = max(n_neg_all / max(n_pos_all, 1), 1.0)

    # 4) Train candidates
    results = {}
    use_pnu = cfg["USE_CONFIDENT_NEGATIVES"] and (df_neg is not None)
    if use_pnu:
        # Logistic
        lr = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=42)
        res_lr = train_eval_PNU(lr, None, df_pos, df_neg, id_cols, ts_col, target_precision)
        results["PNU_LogReg"] = res_lr
        # XGB
        if _HAS_XGB:
            xgb_params = dict(cfg["XGB_PARAMS"])
            xgb = XGBClassifier(**xgb_params)
            res_xgb = train_eval_PNU(xgb, None, df_pos, df_neg, id_cols, ts_col, target_precision)
            results["PNU_XGB"] = res_xgb
        else:
            print("XGBoost not available; skipping.")

    # PU setup
    df_pos_pu = df_pos.copy(); df_pos_pu["is_neuro_labeled"] = 1
    df_unl_pu = df_unl.copy(); df_unl_pu["is_neuro_labeled"] = 0
    df_all_pu = pd.concat([df_pos_pu, df_unl_pu], ignore_index=True)
    num_cols_pu, cat_cols_pu = infer_feature_lists(df_all_pu, id_cols, ["is_neuro_labeled"])
    preproc_pu = make_preprocessor(num_cols_pu, cat_cols_pu)
    Xtr, Xte, ytr, yte = time_or_random_split(df_all_pu, "is_neuro_labeled", ts_col, test_size=0.3, seed=42)

    lr_pu = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=42)
    res_pu_lr = train_eval_PU(lr_pu, preproc_pu, Xtr, ytr, Xte, yte, target_precision)
    results["PU_LogReg"] = res_pu_lr

    if _HAS_XGB:
        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        xgb_params = dict(cfg["XGB_PARAMS"])
        xgb_params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)
        xgb_pu = XGBClassifier(**xgb_params)
        res_pu_xgb = train_eval_PU(xgb_pu, preproc_pu, Xtr, ytr, Xte, yte, target_precision)
        results["PU_XGB"] = res_pu_xgb

    # 5) Select best by meeting precision then highest PR-AUC
    best_name, best_res, best_pr_auc = None, None, -1.0
    for name, res in results.items():
        meets = res["precision_at_tau"] >= target_precision
        print(f"Candidate {name}: PR_AUC={res['pr_auc']:.4f}, Precision@tau={res['precision_at_tau']:.4f}, meets={meets}")
        if meets and res["pr_auc"] > best_pr_auc:
            best_name, best_res, best_pr_auc = name, res, res["pr_auc"]
    if best_res is None:
        print(f"WARNING: No model reached precision ≥ {target_precision:.2f}. Selecting highest PR AUC anyway.")
        for name, res in results.items():
            if res["pr_auc"] > best_pr_auc:
                best_name, best_res, best_pr_auc = name, res, res["pr_auc"]
    print(f"\nSelected model: {best_name} (PR_AUC={best_res['pr_auc']:.4f}, Precision@tau={best_res['precision_at_tau']:.4f}, tau={best_res['tau']:.4f})")

    # 6) Save artifacts
    meta = {"tau": float(best_res["tau"]), "c": float(best_res.get("c", 1.0)), "model": best_name}
    # For PU, optionally recompute c via OOF on combined data
    if best_name and best_name.startswith("PU"):
        pipe_for_c = Pipeline([("prep", preproc_pu), ("clf", best_res["pipeline"].named_steps["clf"])])
        X_all = pd.concat([Xtr, Xte], axis=0)
        y_all = pd.concat([ytr, yte], axis=0)
        oof_c = estimate_oof_c(pipe_for_c, X_all, y_all)
        meta["c"] = float(oof_c)

    joblib.dump(best_res["pipeline"], cfg["ARTIFACT_MODEL"])
    with open(cfg["ARTIFACT_JSON"], "w") as f:
        json.dump(meta, f)
    print(f"Saved pipeline -> {cfg['ARTIFACT_MODEL']}, meta -> {cfg['ARTIFACT_JSON']}")

    # 7) Score unlabeled and write outputs
    score_and_save_unlabeled(best_res, df_unl, id_cols,
                             cfg["OUTPUT_SCORES_CSV"], cfg["OUTPUT_FLAGGED_CSV"])
    print("Done.")

if __name__ == "__main__":
    main()
