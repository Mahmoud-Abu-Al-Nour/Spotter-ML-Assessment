"""Validation procedures: expanding-window cross-validation and holdout evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from src.config import CV_FOLDS, HOLDOUT_SPLIT_DATE
from src.preprocessing import FeaturePipeline
from src.model import build_model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict[str, float]:
    """Calculate and display evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    med_ae = np.median(np.abs(y_true - y_pred))
    print(
        f"  {label:28s} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | "
        f"R2: {r2:.6f} | MAPE: {mape:.2f}% | MedAE: ${med_ae:,.2f}"
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "med_ae": med_ae}


def run_cross_validation(
    train_df: pd.DataFrame,
) -> Tuple[List[Dict[str, float]], List[int], List[str]]:
    """
    Execute expanding-window time-series cross-validation.
    Ensures complete preprocessing isolation inside each training split.
    """
    cv_metrics = []
    best_iters = []
    feature_cols = []

    for name, vstart, vend in CV_FOLDS:
        tr_df = train_df[train_df["date"] < vstart].copy()
        vl_df = train_df[(train_df["date"] >= vstart) & (train_df["date"] < vend)].copy()

        pipeline = FeaturePipeline()
        pipeline.fit(tr_df)
        oof_enc = pipeline.compute_oof_target_encodings(tr_df, n_splits=5)

        X_tr = pipeline.transform(tr_df, is_training=True, oof_encodings=oof_enc)
        y_tr = tr_df["posted_rate"]
        X_vl = pipeline.transform(vl_df, is_training=False)
        y_vl = vl_df["posted_rate"]

        feature_cols = list(X_tr.columns)

        model = build_model(n_estimators=3000)
        model.fit(
            X_tr[feature_cols],
            y_tr,
            eval_set=[(X_vl[feature_cols], y_vl)],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )

        preds = np.clip(model.predict(X_vl[feature_cols]), 1.0, None)
        m = evaluate(y_vl.values, preds, name)
        cv_metrics.append(m)
        best_iters.append(model.best_iteration_)

    return cv_metrics, best_iters, feature_cols


def run_holdout(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Dict[str, float], int]:
    """
    Execute 2-month out-of-time holdout evaluation (Jan-Aug train -> Sep-Oct test).
    Approximates the contiguous 2-month future prediction horizon of Nov-Dec.
    """
    tr_h = train_df[train_df["date"] < HOLDOUT_SPLIT_DATE].copy()
    vl_h = train_df[train_df["date"] >= HOLDOUT_SPLIT_DATE].copy()

    pipeline_h = FeaturePipeline()
    pipeline_h.fit(tr_h)
    oof_enc_h = pipeline_h.compute_oof_target_encodings(tr_h, n_splits=5)

    X_tr_h = pipeline_h.transform(tr_h, is_training=True, oof_encodings=oof_enc_h)
    y_tr_h = tr_h["posted_rate"]
    X_vl_h = pipeline_h.transform(vl_h, is_training=False)
    y_vl_h = vl_h["posted_rate"]

    model_h = build_model(n_estimators=3000)
    model_h.fit(
        X_tr_h[feature_cols],
        y_tr_h,
        eval_set=[(X_vl_h[feature_cols], y_vl_h)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )

    h_preds = np.clip(model_h.predict(X_vl_h[feature_cols]), 1.0, None)
    h_m = evaluate(y_vl_h.values, h_preds, "Sep-Oct Holdout (2 mo)")
    return h_m, model_h.best_iteration_
