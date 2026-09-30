"""Feature engineering transformations for freight load attributes."""

import numpy as np
import pandas as pd

from src.config import EARTH_RADIUS_MILES, EQUIPMENT_CODE_MAP, EQUIPMENT_PREMIUM_MAP


def haversine(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Calculate great-circle distance between coordinate pairs in miles."""
    r = EARTH_RADIUS_MILES
    phi1, lambda1, phi2, lambda2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = phi2 - phi1
    dlon = lambda2 - lambda1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def compute_distance_features(distance: pd.Series) -> pd.DataFrame:
    """Compute primary distance transformations."""
    df = pd.DataFrame(index=distance.index)
    dist_val = distance.astype(float)
    df["distance"] = dist_val
    df["log_distance"] = np.log1p(dist_val)
    df["distance_sq"] = dist_val ** 2
    return df


def compute_weight_features(weight: pd.Series) -> pd.DataFrame:
    """Compute weight transformations."""
    df = pd.DataFrame(index=weight.index)
    w_val = weight.astype(float)
    df["weight"] = w_val
    df["log_weight"] = np.log1p(w_val)
    return df


def compute_equipment_features(equipment: pd.Series) -> pd.DataFrame:
    """Compute equipment code and rate multiplier."""
    df = pd.DataFrame(index=equipment.index)
    df["equipment_code"] = equipment.map(EQUIPMENT_CODE_MAP).fillna(0).astype(int)
    df["equip_premium"] = equipment.map(EQUIPMENT_PREMIUM_MAP).fillna(1.0)
    return df


def compute_calendar_features(date: pd.Series) -> pd.DataFrame:
    """Compute day-of-week, day-of-month, and cyclical sinusoidal encodings."""
    df = pd.DataFrame(index=date.index)
    dt = pd.to_datetime(date)
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["dom_sin"] = np.sin(2.0 * np.pi * df["day_of_month"] / 31.0)
    df["dom_cos"] = np.cos(2.0 * np.pi * df["day_of_month"] / 31.0)
    return df


def compute_spatial_geometry(
    plat: pd.Series,
    plon: pd.Series,
    dlat: pd.Series,
    dlon: pd.Series,
    distance: pd.Series,
) -> pd.DataFrame:
    """Compute geographic vectors, midpoint, great-circle distance, and circuity."""
    df = pd.DataFrame(index=plat.index)
    df["pickup_lat"] = plat.astype(float)
    df["pickup_lon"] = plon.astype(float)
    df["delivery_lat"] = dlat.astype(float)
    df["delivery_lon"] = dlon.astype(float)

    h_dist = haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    df["haversine_dist"] = h_dist
    df["circuity"] = distance.astype(float) / np.clip(h_dist, 1.0, None)
    df["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    df["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
    df["abs_delta_lat"] = np.abs(df["delta_lat"])
    df["abs_delta_lon"] = np.abs(df["delta_lon"])
    df["mid_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2.0
    df["mid_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2.0
    return df


def compute_interaction_features(
    distance: pd.Series,
    log_distance: pd.Series,
    weight: pd.Series,
    equip_premium: pd.Series,
) -> pd.DataFrame:
    """Compute domain interaction features."""
    df = pd.DataFrame(index=distance.index)
    df["dist_x_weight"] = distance * weight
    df["dist_x_premium"] = distance * equip_premium
    df["log_dist_x_premium"] = log_distance * equip_premium
    return df
