# negbank_to_pnu_pipeline.py
import os, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
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

# ======================
# CONFIG
# ======================
CONFIG = {
    # --- Input CSVs ---
    "POS_CSV": "glaucoma_positives.csv",          # confirmed positives (P)
    "NEG_BANK_CSV": "negative_bank_cleaned.csv",    # current negative bank (U for Stage A)
    "UNLABELED_CSV": "glaucoma_suspects.csv",    # large unlabeled pool for Stage B

    # --- Columns ---
    "ID_COLUMNS": ["Patient ID", "Study Instance UID", "Date of Birth", "OCT Scan Date"],
    "PATIENT_COL": "Patient ID",

    # --- PU Stage (build confident negatives from NEG_BANK_CSV) ---
    "PU_TARGET_PRECISION": 0.98,      # very conservative; we want reliable scores
    "PU_FALLBACK_TAU": 0.99,          # if target precision cannot be achieved, use very high tau
    "CONF_NEG_SELECTION": "quantile", # "quantile" or "topk_low" (use bottom scores)
    "CONF_NEG_Q": 0.20,               # keep bottom 20% as confident negatives
    "CONF_NEG_MIN": 3000,             # ensure at least this many confident negatives (if available)

    # --- PNU Stage (supervised P vs ConfNeg) ---
    "PNU_TARGET_PRECISION": 0.95,
    "PNU_FALLBACK_TAU": 0.97,

    # --- XGB params (used if xgboost installed) ---
    "XGB_PARAMS": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42
    },

    # --- Outputs ---
    "CONF_NEG_CSV": "confident_negatives.csv",
    "PU_MODEL_ARTIFACT": "neg_bank_pu_model.joblib",
    "PU_META": "neg_bank_pu_meta.json",

    "PNU_MODEL_ARTIFACT": "pnu_best_model.joblib",
    "PNU_META": "pnu_meta.json",

    "UNLABELED_SCORES_CSV": "unlabeled_scores_pnu.csv",
    "UNLABELED_FLAGGED_CSV": "unlabeled_flagged_pnu.csv",
    "QA_SAMPLE_CSV": "pnu_qa_sample.csv",
}

# ======================
# IO
# ======================
def load_csv(path: str) -> pd.DataFrame:
    assert os.path.isfile(path), f"Missing file: {path}"
    return pd.read_csv(path)

# ======================
# Feature utilities
# ======================
DROP_DATE_COLS = {"Date of Birth", "OCT Scan Date"}

def ignore_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Completely drop raw date columns; do NOT engineer date features.
    df = df.copy()
    for col in DROP_DATE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def add_missing_flags(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    df = df.copy()
    exclude = set(exclude or [])
    rates = df.isna().mean()
    for c, r in rates.items():
        if c not in exclude and r > 0:
            df[f"{c}__isna"] = df[c].isna().astype(int)
    return df

def drop_very_sparse(df: pd.DataFrame, thresh=0.90) -> pd.DataFrame:
    rates = df.isna().mean()
    drop = rates[rates > thresh].index.tolist()
    if drop:
        print(f"Dropping very sparse cols (> {int(thresh*100)}% NaN): {drop}")
        df = df.drop(columns=drop)
    return df

def align_columns(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    # Align all DataFrames to the union of their columns
    all_cols = sorted(set().union(*[set(df.columns) for df in dfs]))
    return tuple(df.reindex(columns=all_cols) for df in dfs)

def split_features(df: pd.DataFrame, id_cols: list, label_col: Optional[str] = None) -> Tuple[list, list]:
    drop_cols = set(id_cols + ([label_col] if label_col else []))
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
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

# ======================
# Splits
# ======================
def group_split(df: pd.DataFrame, label_col: str, patient_col: Optional[str], test_size=0.3, seed=42):
    if patient_col and patient_col in df.columns:
        groups = df[patient_col].fillna("MISSING")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx = np.arange(len(df))
        for tr, te in gss.split(idx, df[label_col], groups):
            return tr, te
    # fallback
    return train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed, stratify=df[label_col])

# ======================
# PU helpers
# ======================
def estimate_c_oof(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, patient_col: Optional[str] = None, seed=42) -> float:
    """Elkan–Noto c = P(s=1|y=1) from OOF predictions on labeled positives."""
    if patient_col and patient_col in X.columns:
        groups = X[patient_col].fillna("MISSING")
        splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        oof = np.zeros(len(X))
        idx = np.arange(len(X))
        for tr, te in splitter.split(idx, y, groups):
            fold = clone(pipe)
            fold.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = fold.predict_proba(X.iloc[te])[:, 1]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        for tr, te in skf.split(X, y):
            fold = clone(pipe)
            fold.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = fold.predict_proba(X.iloc[te])[:, 1]
    c = float(np.clip(oof[y == 1].mean(), 1e-6, 1.0))
    return c

def pick_tau_for_precision(y_true_pos_mask: np.ndarray, scores: np.ndarray, target_precision: float, fallback=0.9) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true_pos_mask, scores)
    for pr, th in zip(precision[:-1], thresholds):
        if pr >= target_precision:
            return float(th)
    return float(fallback)

def eval_pu_model(base_clf, preproc, X_train, y_train, X_test, y_test,
                  target_precision=0.95, fallback_tau=0.99, patient_col=None):
    pipe = Pipeline([("prep", preproc), ("clf", base_clf)])
    pipe.fit(X_train, y_train)

    # robust c via OOF on all data (train+test)
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    c = estimate_c_oof(Pipeline([("prep", preproc), ("clf", clone(base_clf))]),
                       X_all, y_all, patient_col=patient_col)

    ps = pipe.predict_proba(X_test)[:, 1]
    ptrue = np.clip(ps / max(c, 1e-6), 0.0, 1.0)
    tau = pick_tau_for_precision(y_test.values, ptrue, target_precision, fallback=fallback_tau)
    pr_auc = float(average_precision_score(y_test, ptrue))
    y_hat = (ptrue >= tau).astype(int)
    tp = int(((y_test == 1) & (y_hat == 1)).sum())
    pred_pos = int((y_hat == 1).sum())
    actual_pos = int((y_test == 1).sum())
    precision_at_tau = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall_pos = float(tp / actual_pos) if actual_pos > 0 else 0.0
    print(f"[PU] PR_AUC={pr_auc:.4f}  Precision@tau={precision_at_tau:.4f}  Recall_pos={recall_pos:.4f}  tau={tau:.4f}  c={c:.4f}")
    return {"pipeline": pipe, "tau": tau, "c": c, "pr_auc": pr_auc,
            "precision_at_tau": precision_at_tau, "recall_pos": recall_pos}

# ======================
# Stage A: Build confident negatives from NEG_BANK via PU
# ======================
def stageA_build_conf_neg(pos_df: pd.DataFrame, negbank_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # 1) Feature prep (ignore dates, add NA flags, drop sparse)
    pos = ignore_dates(pos_df)
    nb  = ignore_dates(negbank_df)

    pos = add_missing_flags(pos, exclude=cfg["ID_COLUMNS"])
    nb  = add_missing_flags(nb,  exclude=cfg["ID_COLUMNS"])

    pos = drop_very_sparse(pos, 0.90)
    nb  = drop_very_sparse(nb,  0.90)

    # 2) Align columns and build PU frame: s=1 for P, s=0 for bank
    pos, nb = align_columns(pos, nb)
    pos_pu = pos.copy(); pos_pu["is_pos_labeled"] = 1
    nb_pu  = nb.copy();  nb_pu["is_pos_labeled"]  = 0
    df_all = pd.concat([pos_pu, nb_pu], ignore_index=True)

    # 3) Split by patient to avoid leakage
    label_col = "is_pos_labeled"
    tr_idx, te_idx = group_split(df_all, label_col, cfg["PATIENT_COL"], test_size=0.3, seed=42)
    df_tr = df_all.iloc[tr_idx].reset_index(drop=True)
    df_te = df_all.iloc[te_idx].reset_index(drop=True)

    # 4) Preprocessor on union of columns
    num_cols, cat_cols = split_features(df_all, cfg["ID_COLUMNS"], label_col)
    preproc = make_preprocessor(num_cols, cat_cols)

    X_tr = df_tr.drop(columns=[label_col], errors="ignore")
    y_tr = df_tr[label_col]
    X_te = df_te.drop(columns=[label_col], errors="ignore")
    y_te = df_te[label_col]

    # 5) Candidate models
    results = {}
    models = {
        "PU_LogReg": LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    }
    if _HAS_XGB:
        n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
        xgb_params = dict(cfg["XGB_PARAMS"])
        xgb_params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)
        models["PU_XGB"] = XGBClassifier(**xgb_params)

    print("\nTraining models for Stage A (PU on negative bank) ...")
    for name, clf in models.items():
        print(f"\n------------ {name} ------------")
        res = eval_pu_model(
            clf, preproc, X_tr, y_tr, X_te, y_te,
            target_precision=cfg["PU_TARGET_PRECISION"],
            fallback_tau=cfg["PU_FALLBACK_TAU"],
            patient_col=cfg["PATIENT_COL"]
        )
        results[name] = res

    # 6) Pick best that meets precision; tiebreak by PR-AUC
    target = cfg["PU_TARGET_PRECISION"]
    best_name, best_res, best_auc = None, None, -1
    for name, res in results.items():
        meets = res["precision_at_tau"] >= target
        print(f"Candidate {name}: PR_AUC={res['pr_auc']:.4f}, Precision@tau={res['precision_at_tau']:.4f}, meets={meets}")
        if meets and res["pr_auc"] > best_auc:
            best_name, best_res, best_auc = name, res, res["pr_auc"]
    if best_res is None:
        print(f"WARNING: no Stage A model reached precision ≥ {target:.2f}. Selecting highest PR-AUC anyway.")
        for name, res in results.items():
            if res["pr_auc"] > best_auc:
                best_name, best_res, best_auc = name, res, res["pr_auc"]

    print(f"\nSelected Stage A model: {best_name} (PR_AUC={best_res['pr_auc']:.4f}, "
          f"Precision@tau={best_res['precision_at_tau']:.4f}, tau={best_res['tau']:.4f})")

    # 7) Save Stage A artifacts
    joblib.dump(best_res["pipeline"], CONFIG["PU_MODEL_ARTIFACT"])
    with open(CONFIG["PU_META"], "w") as f:
        json.dump({"tau": best_res["tau"], "c": best_res["c"], "features_num": num_cols, "features_cat": cat_cols}, f)
    print(f"Saved Stage A pipeline -> {CONFIG['PU_MODEL_ARTIFACT']}, meta -> {CONFIG['PU_META']}")

    # 8) Score entire negative bank to pick confident negatives (lowest scores)
    pipe, tau, c = best_res["pipeline"], best_res["tau"], best_res["c"]

    X_nb = nb.drop(columns=[c for c in CONFIG["ID_COLUMNS"] if c in nb.columns], errors="ignore")
    # ensure X_nb has the same columns the preprocessor expects: safe because fit on union (df_all)
    ps_nb = pipe.predict_proba(X_nb)[:, 1]
    ptrue_nb = np.clip(ps_nb / max(c, 1e-6), 0.0, 1.0)

    # Select confident negatives: lowest ptrue
    n_total = len(ptrue_nb)
    if CONFIG["CONF_NEG_SELECTION"] == "quantile":
        q = CONFIG["CONF_NEG_Q"]
        cutoff = np.quantile(ptrue_nb, q)
        conf_mask = ptrue_nb <= cutoff
        # enforce minimum count if possible
        if conf_mask.sum() < CONFIG["CONF_NEG_MIN"]:
            order = np.argsort(ptrue_nb)  # ascending
            k = min(CONFIG["CONF_NEG_MIN"], n_total)
            conf_mask = np.zeros(n_total, dtype=bool); conf_mask[order[:k]] = True
    else:  # "topk_low"
        order = np.argsort(ptrue_nb)
        k = min(CONFIG["CONF_NEG_MIN"], n_total)
        conf_mask = np.zeros(n_total, dtype=bool); conf_mask[order[:k]] = True

    conf_neg = negbank_df.loc[conf_mask].copy()
    conf_neg.to_csv(CONFIG["CONF_NEG_CSV"], index=False)
    print(f"Stage A: selected {len(conf_neg)} confident negatives -> {CONFIG['CONF_NEG_CSV']}")

    # Return confident negatives + info we’ll need
    return conf_neg, {"preproc_num_cols": num_cols, "preproc_cat_cols": cat_cols}

# ======================
# Stage B: PNU (supervised P vs ConfNeg) on UNLABELED
# ======================
def stageB_pnu(pos_df: pd.DataFrame, conf_neg_df: pd.DataFrame, unl_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    # 1) Feature prep identical to Stage A (ignore dates, flags, drop sparse)
    pos = ignore_dates(pos_df)
    cn  = ignore_dates(conf_neg_df)
    unl = ignore_dates(unl_df)

    pos = add_missing_flags(pos, exclude=cfg["ID_COLUMNS"])
    cn  = add_missing_flags(cn,  exclude=cfg["ID_COLUMNS"])
    unl = add_missing_flags(unl, exclude=cfg["ID_COLUMNS"])

    pos = drop_very_sparse(pos, 0.90)
    cn  = drop_very_sparse(cn,  0.90)
    unl = drop_very_sparse(unl, 0.90)

    # 2) Align all three (P, ConfNeg, Unlabeled)
    pos, cn, unl = align_columns(pos, cn, unl)

    # 3) Build supervised frame y=1 for P, y=0 for ConfNeg
    y_col = "y_supervised"
    pos_s = pos.copy(); pos_s[y_col] = 1
    cn_s  = cn.copy();  cn_s[y_col]  = 0
    df_sup = pd.concat([pos_s, cn_s], ignore_index=True)

    # 4) Split by patient for evaluation threshold selection
    tr_idx, te_idx = group_split(df_sup, y_col, cfg["PATIENT_COL"], test_size=0.3, seed=42)
    df_tr = df_sup.iloc[tr_idx].reset_index(drop=True)
    df_te = df_sup.iloc[te_idx].reset_index(drop=True)

    # 5) Preprocessor
    num_cols, cat_cols = split_features(df_sup, cfg["ID_COLUMNS"], y_col)
    preproc = make_preprocessor(num_cols, cat_cols)

    # 6) Models
    models = {
        "PNU_LogReg": LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    }
    if _HAS_XGB:
        y_tr = df_tr[y_col]
        n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
        xgb_params = dict(cfg["XGB_PARAMS"])
        xgb_params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)
        models["PNU_XGB"] = XGBClassifier(**xgb_params)

    # 7) Train/eval and choose tau for target precision
    def eval_pnu(base_clf):
        pipe = Pipeline([("prep", preproc), ("clf", base_clf)])
        Xtr = df_tr.drop(columns=[y_col], errors="ignore")
        ytr = df_tr[y_col]
        Xte = df_te.drop(columns=[y_col], errors="ignore")
        yte = df_te[y_col]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        tau = pick_tau_for_precision(yte.values, p, cfg["PNU_TARGET_PRECISION"], fallback=cfg["PNU_FALLBACK_TAU"])
        pr_auc = float(average_precision_score(yte, p))
        yhat = (p >= tau).astype(int)
        tp = int(((yte == 1) & (yhat == 1)).sum())
        pred_pos = int((yhat == 1).sum())
        actual_pos = int((yte == 1).sum())
        precision_at_tau = float(tp / pred_pos) if pred_pos > 0 else 0.0
        recall_pos = float(tp / actual_pos) if actual_pos > 0 else 0.0
        print(f"[PNU] PR_AUC={pr_auc:.4f}  Precision@tau={precision_at_tau:.4f}  Recall_pos={recall_pos:.4f}  tau={tau:.4f}")
        return {"pipeline": pipe, "tau": tau, "pr_auc": pr_auc,
                "precision_at_tau": precision_at_tau, "recall_pos": recall_pos}

    results = {}
    print("\nTraining models for Stage B (PNU on unlabeled) ...")
    for name, clf in models.items():
        print(f"\n------------ {name} ------------")
        results[name] = eval_pnu(clf)

    # Select best meeting precision target
    target = cfg["PNU_TARGET_PRECISION"]
    best_name, best_res, best_auc = None, None, -1
    for name, res in results.items():
        meets = res["precision_at_tau"] >= target
        print(f"Candidate {name}: PR_AUC={res['pr_auc']:.4f}, Precision@tau={res['precision_at_tau']:.4f}, meets={meets}")
        if meets and res["pr_auc"] > best_auc:
            best_name, best_res, best_auc = name, res, res["pr_auc"]
    if best_res is None:
        print(f"WARNING: no Stage B model reached precision ≥ {target:.2f}. Selecting highest PR-AUC anyway.")
        for name, res in results.items():
            if res["pr_auc"] > best_auc:
                best_name, best_res, best_auc = name, res, res["pr_auc"]

    print(f"\nSelected Stage B model: {best_name} (PR_AUC={best_res['pr_auc']:.4f}, "
          f"Precision@tau={best_res['precision_at_tau']:.4f}, tau={best_res['tau']:.4f})")

    # Save Stage B artifacts
    joblib.dump(best_res["pipeline"], CONFIG["PNU_MODEL_ARTIFACT"])
    with open(CONFIG["PNU_META"], "w") as f:
        json.dump({"tau": best_res["tau"], "features_num": num_cols, "features_cat": cat_cols}, f)
    print(f"Saved Stage B pipeline -> {CONFIG['PNU_MODEL_ARTIFACT']}, meta -> {CONFIG['PNU_META']}")

    # Score unlabeled
    X_unl = unl.drop(columns=[c for c in CONFIG["ID_COLUMNS"] if c in unl.columns], errors="ignore")
    ps_unl = best_res["pipeline"].predict_proba(X_unl)[:, 1]
    flags = (ps_unl >= best_res["tau"]).astype(int)

    keep_ids = [c for c in CONFIG["ID_COLUMNS"] if c in unl_df.columns]
    scores_df = pd.DataFrame({**{col: unl_df[col] for col in keep_ids}, "p_pos": ps_unl, "flag_poslike": flags})
    scores_df.to_csv(CONFIG["UNLABELED_SCORES_CSV"], index=False)
    unl_df.loc[flags == 1].to_csv(CONFIG["UNLABELED_FLAGGED_CSV"], index=False)
    print(f"Saved scores: {CONFIG['UNLABELED_SCORES_CSV']}  |  Flagged rows: {(flags==1).sum()} -> {CONFIG['UNLABELED_FLAGGED_CSV']}")

    # QA sample: 20 high, 20 near tau, 20 low
    df_scores = scores_df.copy()
    df_scores["delta"] = (df_scores["p_pos"] - best_res["tau"]).abs()
    high = df_scores[df_scores["p_pos"] >= min(best_res["tau"] + 0.1, 1.0)].nlargest(20, "p_pos")
    around = df_scores.nsmallest(20, "delta")
    low = df_scores[df_scores["p_pos"] <= max(best_res["tau"] - 0.1, 0.0)].nsmallest(20, "p_pos")
    qa_ids = pd.concat([high, around, low], ignore_index=True)[keep_ids]
    qa = qa_ids.merge(unl_df, on=keep_ids, how="left")
    qa.to_csv(CONFIG["QA_SAMPLE_CSV"], index=False)
    print(f"Saved QA sample -> {CONFIG['QA_SAMPLE_CSV']}")

# ======================
# MAIN
# ======================
def main(cfg=CONFIG):
    # Load
    pos = load_csv(cfg["POS_CSV"])
    negbank = load_csv(cfg["NEG_BANK_CSV"])
    unlabeled = load_csv(cfg["UNLABELED_CSV"])

    # Stage A: PU on negative bank -> confident negatives
    conf_neg, _ = stageA_build_conf_neg(pos, negbank, cfg)

    # Stage B: PNU (supervised) using P vs confident negatives -> score unlabeled
    stageB_pnu(pos, conf_neg, unlabeled, cfg)

if __name__ == "__main__":
    main()
