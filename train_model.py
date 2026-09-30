"""Main pipeline orchestration for freight rate prediction."""

import subprocess
import sys
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEMPLATE_PATH,
    DECEMBER_PATH,
    VAL_OUTPUT_PATH,
    DEC_OUTPUT_PATH,
)
from src.preprocessing import FeaturePipeline
from src.model import train_final_model
from src.validation import evaluate, run_cross_validation, run_holdout
from src.predict import (
    generate_validation_predictions,
    generate_december_predictions,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw development, validation, template, and December benchmark data."""
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    template_df = pd.read_csv(TEMPLATE_PATH)
    dec_df = pd.read_csv(DECEMBER_PATH)

    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"] = pd.to_datetime(val_df["date"])
    dec_df["date"] = pd.to_datetime(dec_df["date"])
    return train_df, val_df, template_df, dec_df


def main() -> None:
    print("Starting freight rate model training pipeline...")
    train_df, val_df, template_df, dec_df = load_data()
    print(f"Loaded datasets: Train={train_df.shape}, Val={val_df.shape}, Dec={dec_df.shape}")

    # 1. Expanding-window time-series cross-validation
    print("\n--- Running Time-Series Cross-Validation ---")
    cv_metrics, best_iters, feature_cols = run_cross_validation(train_df)
    avg_mae = np.mean([m["mae"] for m in cv_metrics])
    avg_rmse = np.mean([m["rmse"] for m in cv_metrics])
    avg_r2 = np.mean([m["r2"] for m in cv_metrics])
    avg_mape = np.mean([m["mape"] for m in cv_metrics])
    avg_medae = np.mean([m["med_ae"] for m in cv_metrics])
    print(
        f"\n  {'CV 3-Fold Average':28s} | MAE: ${avg_mae:,.2f} | RMSE: ${avg_rmse:,.2f} | "
        f"R2: {avg_r2:.6f} | MAPE: {avg_mape:.2f}% | MedAE: ${avg_medae:,.2f}"
    )

    # 2. 2-month out-of-time holdout evaluation
    print("\n--- Running 2-Month Out-of-Time Holdout (Sep-Oct) ---")
    _, holdout_iter = run_holdout(train_df, feature_cols)
    best_iters.append(holdout_iter)

    # 3. Train final model on full development dataset
    print("\n--- Training Final Model on Full Development Data (48,000 loads) ---")
    final_pipeline = FeaturePipeline()
    final_pipeline.fit(train_df)
    oof_enc_final = final_pipeline.compute_oof_target_encodings(train_df, n_splits=5)

    X_final = final_pipeline.transform(train_df, is_training=True, oof_encodings=oof_enc_final)
    y_final = train_df["posted_rate"]

    n_trees = int(np.mean(best_iters) * 1.25)
    print(f"Fitting final LightGBM model with {n_trees} estimators...")
    final_model = train_final_model(X_final, y_final, feature_cols, n_estimators=n_trees)

    # Sanity evaluation on training set
    train_preds = np.clip(final_model.predict(X_final[feature_cols]), 1.0, None)
    evaluate(y_final.values, train_preds, "Full Train Sanity")

    # 4. Generate validation predictions
    print("\n--- Generating Validation Predictions (12,000 loads) ---")
    generate_validation_predictions(
        final_model, final_pipeline, val_df, template_df, feature_cols, VAL_OUTPUT_PATH
    )

    # 5. Generate December fixed-corridor benchmark predictions
    print("\n--- Generating December Predictions ---")
    generate_december_predictions(
        final_model, final_pipeline, dec_df, feature_cols, DEC_OUTPUT_PATH
    )

    # 6. Run official submission validation
    print("\n--- Executing Official score.py Validation ---")
    subprocess.run(
        [
            sys.executable,
            "score.py",
            "--predictions",
            str(VAL_OUTPUT_PATH),
            "--december-predictions",
            str(DEC_OUTPUT_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
