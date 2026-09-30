"""Prediction generation and output formatting routines."""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List

from src.preprocessing import FeaturePipeline


def generate_validation_predictions(
    model: lgb.LGBMRegressor,
    pipeline: FeaturePipeline,
    val_df: pd.DataFrame,
    template_df: pd.DataFrame,
    feature_cols: List[str],
    output_path: Path | str,
) -> pd.DataFrame:
    """Generate and save validation predictions matching the official submission template."""
    X_val = pipeline.transform(val_df, is_training=False)
    val_preds = np.clip(model.predict(X_val[feature_cols]), 1.0, None)

    output = template_df.copy()
    output["predicted_rate"] = val_preds
    output.to_csv(output_path, index=False)

    # Verification checks
    assert len(output) == 12000, f"Expected 12,000 rows, found {len(output)}"
    assert list(output.columns) == ["load_id", "predicted_rate"], f"Invalid columns: {list(output.columns)}"
    assert output["predicted_rate"].isna().sum() == 0, "Missing values detected"
    assert (output["predicted_rate"] > 0).all(), "Non-positive rates detected"

    print(
        f"  Saved: {output_path}\n"
        f"  Rows: {len(output):,} | Min: ${val_preds.min():,.2f} | "
        f"Max: ${val_preds.max():,.2f} | Mean: ${val_preds.mean():,.2f}"
    )
    return output


def generate_december_predictions(
    model: lgb.LGBMRegressor,
    pipeline: FeaturePipeline,
    december_df: pd.DataFrame,
    feature_cols: List[str],
    output_path: Path | str,
) -> pd.DataFrame:
    """Generate and save December fixed-route daily predictions via direct model inference."""
    X_dec = pipeline.transform(december_df, is_training=False)
    dec_preds = np.clip(model.predict(X_dec[feature_cols]), 1.0, None)

    output = december_df.copy()
    output["predicted_rate"] = dec_preds
    output.to_csv(output_path, index=False)

    # Verification checks
    assert len(output) == 31, f"Expected 31 rows, found {len(output)}"
    expected_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert list(output.columns) == expected_cols, f"Invalid columns: {list(output.columns)}"
    assert output["predicted_rate"].isna().sum() == 0, "Missing values detected"
    assert (output["predicted_rate"] > 0).all(), "Non-positive rates detected"

    print(
        f"  Saved: {output_path}\n"
        f"  Rows: {len(output)} | Min: ${dec_preds.min():,.2f} | "
        f"Max: ${dec_preds.max():,.2f} | Spread: ${dec_preds.max() - dec_preds.min():,.2f}"
    )
    return output
