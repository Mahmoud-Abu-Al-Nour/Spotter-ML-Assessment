"""Model definition and training routines using LightGBM."""

from typing import Dict, Any, List, Optional
import lightgbm as lgb
import pandas as pd

from src.config import MODEL_PARAMS


def build_model(
    n_estimators: Optional[int] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMRegressor:
    """Instantiate a LightGBM regressor with assessment configuration."""
    params = MODEL_PARAMS.copy()
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if custom_params is not None:
        params.update(custom_params)
    return lgb.LGBMRegressor(**params)


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    n_estimators: int = 197,
) -> lgb.LGBMRegressor:
    """Train the final LightGBM model on the complete development dataset."""
    model = build_model(n_estimators=n_estimators)
    model.fit(X_train[feature_cols], y_train)
    return model
