"""Configuration and constants for freight rate prediction."""

from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

# Data file paths
TRAIN_PATH = BASE_DIR / "train-test.csv"
VAL_PATH = BASE_DIR / "validation.csv"
TEMPLATE_PATH = BASE_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = BASE_DIR / "december-chart-inputs.csv"

# Output paths
VAL_OUTPUT_PATH = BASE_DIR / "validation_predictions.csv"
DEC_OUTPUT_PATH = BASE_DIR / "december_predictions.csv"
SCORER_RESULTS_DIR = BASE_DIR / "scorer_results"

RANDOM_SEED = 42

# Target encoding smoothing parameter
TARGET_SMOOTHING = 15.0

# Physical and categorical mappings
EARTH_RADIUS_MILES = 3958.8

EQUIPMENT_CODE_MAP = {
    "Dry Van": 0,
    "Flatbed": 1,
    "Reefer": 2,
}

EQUIPMENT_PREMIUM_MAP = {
    "Dry Van": 1.0,
    "Flatbed": 1.08,
    "Reefer": 1.12,
}

# Cross-validation folds (expanding window)
CV_FOLDS: List[Tuple[str, str, str]] = [
    ("Jan-Jul -> Aug", "2025-08-01", "2025-09-01"),
    ("Jan-Aug -> Sep", "2025-09-01", "2025-10-01"),
    ("Jan-Sep -> Oct", "2025-10-01", "2025-11-01"),
]

# Holdout validation split
HOLDOUT_SPLIT_DATE = "2025-09-01"

# LightGBM hyperparameters
MODEL_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.03,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 197,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}
