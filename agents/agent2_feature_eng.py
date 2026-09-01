"""
Agent 2: feature engineering agent.

Job: take the raw dataset and produce a better feature set for modeling.
Transforms are config-driven so the feedback loop can add new ones between
passes without touching this file.

Output: an engineered DataFrame (as parquet/csv) plus a JSON change log
consumed by Agent 3.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _missing_value_imputation(X: pd.DataFrame, log: list) -> pd.DataFrame:
    for col in X.columns:
        n_missing = X[col].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            median = X[col].median()
            X[col] = X[col].fillna(median)
            log.append(f"{col}: filled {n_missing} missing values with median ({median})")
        else:
            X[col] = X[col].fillna("missing")
            log.append(f"{col}: filled {n_missing} missing values with 'missing' category")
    return X


def _categorical_encoding(X: pd.DataFrame, log: list) -> pd.DataFrame:
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 10:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            log.append(f"{col}: one-hot encoded ({n_unique} categories)")
        else:
            X[col] = LabelEncoder().fit_transform(X[col])
            log.append(f"{col}: label encoded ({n_unique} categories, too many for one-hot)")
    return X


def _numeric_scaling(X: pd.DataFrame, log: list) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=[np.number]).columns
    # bool columns from one-hot encoding shouldn't be scaled
    num_cols = [c for c in num_cols if X[c].dtype != bool and X[c].nunique() > 2]
    if len(num_cols) == 0:
        return X
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    log.append(f"scaled {len(num_cols)} numeric columns with StandardScaler: {list(num_cols)}")
    return X


def _interaction_terms(X: pd.DataFrame, log: list) -> pd.DataFrame:
    if "age" in X.columns and "education_num" in X.columns:
        X["age_x_education"] = X["age"] * X["education_num"]
        log.append("created age_x_education interaction term")
    if "capital_gain" in X.columns and "capital_loss" in X.columns:
        X["net_capital"] = X["capital_gain"] - X["capital_loss"]
        log.append("created net_capital (capital_gain - capital_loss)")
    return X


def _aggregated_features(X: pd.DataFrame, log: list) -> pd.DataFrame:
    if "hours_per_week" in X.columns and "age" in X.columns:
        X["hours_per_age"] = X["hours_per_week"] / X["age"].replace(0, 1)
        log.append("created hours_per_age ratio feature")
    return X


def _domain_specific_transformations(X: pd.DataFrame, log: list) -> pd.DataFrame:
    if "capital_gain" in X.columns:
        X["has_capital_gain"] = (X["capital_gain"] > 0).astype(int)
        log.append("created has_capital_gain binary flag")
    if "capital_loss" in X.columns:
        X["has_capital_loss"] = (X["capital_loss"] > 0).astype(int)
        log.append("created has_capital_loss binary flag")
    return X


TRANSFORM_REGISTRY = {
    # order matters: imputation and encoding must run before scaling/interactions
    "missing_value_imputation": _missing_value_imputation,
    "interaction_terms": _interaction_terms,
    "aggregated_features": _aggregated_features,
    "domain_specific_transformations": _domain_specific_transformations,
    "categorical_encoding": _categorical_encoding,
    "numeric_scaling": _numeric_scaling,
}

# Fixed application order regardless of config list order, so dependencies
# (e.g. imputation before encoding) are always respected.
APPLICATION_ORDER = [
    "missing_value_imputation",
    "interaction_terms",
    "aggregated_features",
    "domain_specific_transformations",
    "categorical_encoding",
    "numeric_scaling",
]


def run(df: pd.DataFrame, target: str, transforms: list[str],
        output_dir: str = "outputs/pass_1") -> dict:
    """
    Applies the requested transforms (in dependency-safe order) and writes
    the engineered dataset + change log.
    Returns a summary dict for Agent 3.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    log = []
    shape_before = X.shape

    ordered_transforms = [t for t in APPLICATION_ORDER if t in transforms]
    skipped = [t for t in transforms if t not in TRANSFORM_REGISTRY]
    for t in skipped:
        log.append(f"[skipped] '{t}' not in transform registry")

    for name in ordered_transforms:
        X = TRANSFORM_REGISTRY[name](X, log)

    shape_after = X.shape

    engineered = X.copy()
    engineered[target] = y.values
    engineered_path = output_path / "engineered_features.csv"
    engineered.to_csv(engineered_path, index=False)

    summary = {
        "transforms_requested": transforms,
        "transforms_applied": ordered_transforms,
        "transforms_skipped": skipped,
        "shape_before": list(shape_before),
        "shape_after": list(shape_after),
        "n_features_added": shape_after[1] - shape_before[1],
        "change_log": log,
        "output_path": str(engineered_path),
    }

    with open(output_path / "agent2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from orchestrator.data_utils import load_config, load_raw_data

    config = load_config()
    df = load_raw_data(config)
    target = config["dataset"]["target"]

    print(f"Running Agent 2 standalone on {df.shape[0]} rows, {df.shape[1] - 1} raw features\n")
    summary = run(
        df, target,
        transforms=config["agent2_feature_engineering"]["transforms"],
        output_dir=f"outputs/pass_{config['pass_number']}",
    )
    print("Change log:")
    for line in summary["change_log"]:
        print(f"  - {line}")
    print(f"\nShape: {summary['shape_before']} -> {summary['shape_after']}")
    print(f"Saved to: {summary['output_path']}")
