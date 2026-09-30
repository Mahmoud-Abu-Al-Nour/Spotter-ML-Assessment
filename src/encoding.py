"""Out-of-fold target encoding with Bayesian smoothing."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Dict, Tuple, Any

from src.config import RANDOM_SEED, TARGET_SMOOTHING


def compute_oof_target_encodings(
    train_df: pd.DataFrame,
    smoothing: float = TARGET_SMOOTHING,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute out-of-fold target encodings on training data.
    Ensures a load's posted_rate never enters its own encoded feature representation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pickup_oof = np.zeros(len(train_df))
    delivery_oof = np.zeros(len(train_df))
    equip_rpm_oof = np.zeros(len(train_df))
    rpm = train_df["posted_rate"] / train_df["distance"].clip(lower=1.0)

    for tr_idx, val_idx in kf.split(train_df):
        tr_split = train_df.iloc[tr_idx]
        val_split = train_df.iloc[val_idx]
        tr_rpm = rpm.iloc[tr_idx]

        g_rate = tr_split["posted_rate"].mean()
        g_rpm = tr_rpm.mean()

        # Pickup target encoding
        p_grp = tr_split.groupby("pickup").agg(s=("posted_rate", "sum"), c=("posted_rate", "count"))
        p_map = ((p_grp["s"] + smoothing * g_rate) / (p_grp["c"] + smoothing)).to_dict()
        pickup_oof[val_idx] = val_split["pickup"].map(p_map).fillna(g_rate).values

        # Delivery target encoding
        d_grp = tr_split.groupby("delivery").agg(s=("posted_rate", "sum"), c=("posted_rate", "count"))
        d_map = ((d_grp["s"] + smoothing * g_rate) / (d_grp["c"] + smoothing)).to_dict()
        delivery_oof[val_idx] = val_split["delivery"].map(d_map).fillna(g_rate).values

        # Equipment RPM target encoding
        e_df = tr_split.assign(_rpm=tr_rpm)
        e_grp = e_df.groupby("equipment").agg(s=("_rpm", "sum"), c=("_rpm", "count"))
        e_map = ((e_grp["s"] + smoothing * g_rpm) / (e_grp["c"] + smoothing)).to_dict()
        equip_rpm_oof[val_idx] = val_split["equipment"].map(e_map).fillna(g_rpm).values

    return pickup_oof, delivery_oof, equip_rpm_oof


def fit_target_encoding_stats(
    train_df: pd.DataFrame,
    smoothing: float = TARGET_SMOOTHING,
) -> Dict[str, Any]:
    """Fit full training target encoding statistics for inference on future data."""
    rpm = train_df["posted_rate"] / train_df["distance"].clip(lower=1.0)
    global_rate_mean = float(train_df["posted_rate"].mean())
    global_rpm_mean = float(rpm.mean())

    p_grp = train_df.groupby("pickup").agg(sum_rate=("posted_rate", "sum"), count=("posted_rate", "count"))
    pickup_stats = {
        c: float((row["sum_rate"] + smoothing * global_rate_mean) / (row["count"] + smoothing))
        for c, row in p_grp.iterrows()
    }

    d_grp = train_df.groupby("delivery").agg(sum_rate=("posted_rate", "sum"), count=("posted_rate", "count"))
    delivery_stats = {
        c: float((row["sum_rate"] + smoothing * global_rate_mean) / (row["count"] + smoothing))
        for c, row in d_grp.iterrows()
    }

    e_df = train_df.assign(_rpm=rpm)
    e_grp = e_df.groupby("equipment").agg(sum_rpm=("_rpm", "sum"), count=("_rpm", "count"))
    equip_stats = {
        c: float((row["sum_rpm"] + smoothing * global_rpm_mean) / (row["count"] + smoothing))
        for c, row in e_grp.iterrows()
    }

    return {
        "pickup_stats": pickup_stats,
        "delivery_stats": delivery_stats,
        "equip_stats": equip_stats,
        "global_rate_mean": global_rate_mean,
        "global_rpm_mean": global_rpm_mean,
    }


def apply_target_encoding(
    df: pd.DataFrame,
    stats: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Apply fitted training target statistics with fallback to global means for unseen entities."""
    pickup_enc = df["pickup"].map(stats["pickup_stats"]).fillna(stats["global_rate_mean"])
    delivery_enc = df["delivery"].map(stats["delivery_stats"]).fillna(stats["global_rate_mean"])
    equip_enc = df["equipment"].map(stats["equip_stats"]).fillna(stats["global_rpm_mean"])
    return pickup_enc, delivery_enc, equip_enc
