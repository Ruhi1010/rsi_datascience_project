"""
Agent 1: model benchmarking agent.

Job: fit a broad set of algorithms on the (already feature-engineered)
dataset and answer:
  - Which algorithms perform best?
  - Which are unstable across folds?
  - Which need better preprocessing?
  - Which deserve deeper tuning next pass?

Output: a leaderboard (CSV) plus a JSON summary consumed by Agent 3.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

MODEL_DEFAULTS = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "decision_tree": (DecisionTreeClassifier, {"random_state": 42}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    "extra_trees": (ExtraTreesClassifier, {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    "gradient_boosting": (GradientBoostingClassifier, {"random_state": 42}),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, {"random_state": 42}),
    "adaboost": (AdaBoostClassifier, {"random_state": 42}),
    "knn": (KNeighborsClassifier, {"n_neighbors": 15}),
    "naive_bayes": (GaussianNB, {}),
}

try:
    from xgboost import XGBClassifier
    MODEL_DEFAULTS["xgboost"] = (XGBClassifier, {"n_estimators": 200, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    MODEL_DEFAULTS["lightgbm"] = (LGBMClassifier, {"n_estimators": 200, "random_state": 42, "verbose": -1})
except ImportError:
    pass


def build_model(name: str, tuned_params: dict | None = None):
    """
    Instantiates a model by name, merging its defaults with any tuned
    hyperparameters carried forward from Agent 5's search on a prior pass.
    svm_linear is handled separately since it's a wrapped estimator.
    """
    if name == "svm_linear":
        return CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3)
    cls, defaults = MODEL_DEFAULTS[name]
    params = {**defaults, **(tuned_params or {})}
    return cls(**params)


# Kept for backward compatibility / introspection (e.g. "name in registry" checks)
MODEL_REGISTRY = {name: (lambda n=name: build_model(n)) for name in MODEL_DEFAULTS}
MODEL_REGISTRY["svm_linear"] = lambda: build_model("svm_linear")


def run(X: pd.DataFrame, y: pd.Series, model_families: list[str], cv_folds: int = 5,
        scoring: str = "roc_auc", output_dir: str = "outputs/pass_1",
        tuned_params: dict | None = None) -> dict:
    """
    Benchmarks the requested model families with stratified k-fold CV.
    If tuned_params is provided (model_name -> hyperparameter dict, carried
    forward from Agent 5's search on a prior pass), those models are
    benchmarked with tuned hyperparameters instead of defaults.
    Returns a summary dict and writes leaderboard.csv + agent1_summary.json.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tuned_params = tuned_params or {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name in model_families:
        if name not in MODEL_REGISTRY:
            print(f"  [skip] '{name}' not in model registry")
            continue

        model = build_model(name, tuned_params.get(name))
        used_tuned = name in tuned_params
        start = time.time()
        try:
            cv_result = cross_validate(
                model, X, y, cv=cv, scoring=scoring,
                return_train_score=True, n_jobs=-1, error_score="raise",
            )
            elapsed = time.time() - start
            test_scores = cv_result["test_score"]
            train_scores = cv_result["train_score"]

            results.append({
                "model": name,
                "mean_test_score": float(np.mean(test_scores)),
                "std_test_score": float(np.std(test_scores)),
                "mean_train_score": float(np.mean(train_scores)),
                "overfit_gap": float(np.mean(train_scores) - np.mean(test_scores)),
                "fit_time_sec": round(elapsed, 2),
                "used_tuned_params": used_tuned,
                "status": "ok",
            })
            tag = " (tuned)" if used_tuned else ""
            print(f"  [ok] {name:20s}{tag} {scoring}={np.mean(test_scores):.4f} "
                  f"(+/- {np.std(test_scores):.4f})  {elapsed:.1f}s")
        except Exception as e:
            results.append({
                "model": name,
                "mean_test_score": None,
                "std_test_score": None,
                "mean_train_score": None,
                "overfit_gap": None,
                "fit_time_sec": None,
                "used_tuned_params": used_tuned,
                "status": f"error: {str(e)[:120]}",
            })
            print(f"  [fail] {name:20s} {str(e)[:100]}")

    leaderboard = pd.DataFrame(results).sort_values(
        "mean_test_score", ascending=False, na_position="last"
    )
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)

    ok = leaderboard[leaderboard["status"] == "ok"]
    unstable = ok[ok["std_test_score"] > ok["std_test_score"].median() * 1.5]["model"].tolist() if len(ok) else []
    overfitting = ok[ok["overfit_gap"] > 0.05]["model"].tolist() if len(ok) else []
    top_models = ok.head(5)["model"].tolist() if len(ok) else []

    summary = {
        "scoring_metric": scoring,
        "cv_folds": cv_folds,
        "n_models_attempted": len(model_families),
        "n_models_succeeded": int((leaderboard["status"] == "ok").sum()),
        "top_models": top_models,
        "unstable_models": unstable,
        "overfitting_models": overfitting,
        "best_model": top_models[0] if top_models else None,
        "best_score": float(ok.iloc[0]["mean_test_score"]) if len(ok) else None,
    }

    with open(output_path / "agent1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from orchestrator.data_utils import load_config, load_raw_data

    config = load_config()
    df = load_raw_data(config)
    target = config["dataset"]["target"]
    positive_class = config["dataset"]["positive_class"]

    # Minimal inline preprocessing so Agent 1 can run standalone for testing.
    # In the full pipeline this comes from Agent 2's output instead.
    from sklearn.preprocessing import LabelEncoder

    X = df.drop(columns=[target]).copy()
    y = (df[target] == positive_class).astype(int)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("missing")
            X[col] = LabelEncoder().fit_transform(X[col])

    print(f"Running Agent 1 standalone on {X.shape[0]} rows, {X.shape[1]} features\n")
    summary = run(
        X, y,
        model_families=config["agent1_benchmarking"]["model_families"],
        cv_folds=config["agent1_benchmarking"]["cv_folds"],
        scoring=config["agent1_benchmarking"]["scoring"],
        output_dir=f"outputs/pass_{config['pass_number']}",
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
