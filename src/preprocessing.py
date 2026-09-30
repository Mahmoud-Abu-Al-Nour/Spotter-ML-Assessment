"""Preprocessing and feature transformation pipeline."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

from src.config import TARGET_SMOOTHING
from src.features import (
    compute_distance_features,
    compute_weight_features,
    compute_equipment_features,
    compute_calendar_features,
    compute_spatial_geometry,
    compute_interaction_features,
)
from src.encoding import (
    compute_oof_target_encodings,
    fit_target_encoding_stats,
    apply_target_encoding,
)


class FeaturePipeline:
    """
    Feature engineering and preprocessing pipeline.
    Parameters are fitted strictly on the training partition and frozen during inference.
    """

    def __init__(self, smoothing: float = TARGET_SMOOTHING):
        self.smoothing = smoothing
        self.weight_imputer: Dict[str, float] = {}
        self.global_weight_median: float = 32000.0
        self.city_coords: Dict[str, Tuple[float, float]] = {}
        self.global_lat_median: float = 39.5
        self.global_lon_median: float = -86.0
        self.encoding_stats: Dict[str, Any] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit preprocessing parameters and global statistics on training data only."""
        clean_weights = train_df["weight"].abs()
        self.global_weight_median = float(clean_weights.median())
        self.weight_imputer = (
            train_df.assign(clean_w=clean_weights)
            .groupby("equipment")["clean_w"]
            .median()
            .to_dict()
        )

        pickup_lats = train_df.groupby("pickup")["pickup_lat"].median().to_dict()
        pickup_lons = train_df.groupby("pickup")["pickup_lon"].median().to_dict()
        delivery_lats = train_df.groupby("delivery")["delivery_lat"].median().to_dict()
        delivery_lons = train_df.groupby("delivery")["delivery_lon"].median().to_dict()

        all_cities = set(list(pickup_lats.keys()) + list(delivery_lats.keys()))
        for city in all_cities:
            lat = pickup_lats.get(city, delivery_lats.get(city))
            lon = pickup_lons.get(city, delivery_lons.get(city))
            if lat is not None and lon is not None:
                self.city_coords[city] = (float(lat), float(lon))

        self.global_lat_median = float(train_df["pickup_lat"].median())
        self.global_lon_median = float(train_df["pickup_lon"].median())

        self.encoding_stats = fit_target_encoding_stats(train_df, smoothing=self.smoothing)

    def compute_oof_target_encodings(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute out-of-fold target encodings on training data."""
        return compute_oof_target_encodings(train_df, smoothing=self.smoothing, n_splits=n_splits)

    def transform(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        oof_encodings: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Transform raw data into model features using frozen preprocessor parameters.
        No statistics are computed from input df during inference.
        """
        f = pd.DataFrame(index=df.index)

        # Distance features
        dist_df = compute_distance_features(df["distance"])
        for col in dist_df.columns:
            f[col] = dist_df[col]

        # Weight features with equipment median imputation
        clean_w = df["weight"].abs()
        imputed_w = clean_w.copy()
        if "equipment" in df.columns:
            for eq, med in self.weight_imputer.items():
                mask = (df["equipment"] == eq) & imputed_w.isna()
                imputed_w[mask] = med
        imputed_w = imputed_w.fillna(self.global_weight_median)
        weight_df = compute_weight_features(imputed_w)
        for col in weight_df.columns:
            f[col] = weight_df[col]

        # Equipment mapping
        equip_df = compute_equipment_features(df["equipment"])
        for col in equip_df.columns:
            f[col] = equip_df[col]

        # Temporal features
        cal_df = compute_calendar_features(df["date"])
        for col in cal_df.columns:
            f[col] = cal_df[col]

        # Geographic coordinates
        plat = df["pickup_lat"].copy() if "pickup_lat" in df.columns else pd.Series(index=df.index, dtype=float)
        plon = df["pickup_lon"].copy() if "pickup_lon" in df.columns else pd.Series(index=df.index, dtype=float)
        dlat = df["delivery_lat"].copy() if "delivery_lat" in df.columns else pd.Series(index=df.index, dtype=float)
        dlon = df["delivery_lon"].copy() if "delivery_lon" in df.columns else pd.Series(index=df.index, dtype=float)

        if plat.isna().any():
            for idx, city in df["pickup"].items():
                if pd.isna(plat.get(idx, np.nan)):
                    coords = self.city_coords.get(city, (self.global_lat_median, self.global_lon_median))
                    plat.loc[idx] = coords[0]
                    plon.loc[idx] = coords[1]

        if dlat.isna().any():
            for idx, city in df["delivery"].items():
                if pd.isna(dlat.get(idx, np.nan)):
                    coords = self.city_coords.get(city, (self.global_lat_median, self.global_lon_median))
                    dlat.loc[idx] = coords[0]
                    dlon.loc[idx] = coords[1]

        geo_df = compute_spatial_geometry(plat, plon, dlat, dlon, df["distance"])
        for col in geo_df.columns:
            f[col] = geo_df[col]

        # Domain interactions
        inter_df = compute_interaction_features(f["distance"], f["log_distance"], f["weight"], f["equip_premium"])
        for col in inter_df.columns:
            f[col] = inter_df[col]

        # Route target encodings
        if is_training and oof_encodings is not None:
            f["pickup_target_enc"] = oof_encodings[0]
            f["delivery_target_enc"] = oof_encodings[1]
            f["equip_rpm_enc"] = oof_encodings[2]
        else:
            p_enc, d_enc, e_enc = apply_target_encoding(df, self.encoding_stats)
            f["pickup_target_enc"] = p_enc
            f["delivery_target_enc"] = d_enc
            f["equip_rpm_enc"] = e_enc

        # Baseline linear rate
        f["est_baseline_rate"] = f["distance"] * f["equip_rpm_enc"]

        return f
