"""
Agent 5: tuning agent.

Job: take the top models flagged by Agent 3 as worth deeper tuning and run
a randomized hyperparameter search on each. Reports whether tuning beats
the untuned baseline from Agent 1. If it does, the winning hyperparameters
are surfaced so Agent 4 can carry them into next_pass_config, and Agent 1
benchmarks that model pre-tuned (instead of with defaults) on the next pass.

This is what makes "deepen hyperparameter tuning for top performers" (an
Agent 3 recommendation) an actual action instead of a description.
"""

import json
from pathlib import Path

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.agent1_benchmarking import MODEL_REGISTRY, build_model

PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "n_estimators": randint(100, 300),
        "max_depth": randint(3, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    },
    "extra_trees": {
        "n_estimators": randint(100, 300),
        "max_depth": randint(3, 20),
        "min_samples_leaf": randint(1, 10),
    },
    "gradient_boosting": {
        "n_estimators": randint(50, 150),
        "max_depth": randint(2, 5),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
    },
    "hist_gradient_boosting": {
        "max_iter": randint(50, 200),
        "max_depth": randint(3, 12),
        "learning_rate": uniform(0.01, 0.29),
        "l2_regularization": uniform(0.0, 1.0),
    },
    "xgboost": {
        "n_estimators": randint(50, 200),
        "max_depth": randint(2, 10),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "lightgbm": {
        "n_estimators": randint(50, 200),
        "num_leaves": randint(15, 127),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
    },
    "logistic_regression": {
        "C": uniform(0.01, 10),
    },
    "decision_tree": {
        "max_depth": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
    },
    "adaboost": {
        "n_estimators": randint(50, 150),
        "learning_rate": uniform(0.1, 1.9),
    },
    "knn": {
        "n_neighbors": randint(3, 50),
        "weights": ["uniform", "distance"],
    },
}


def _to_jsonable(value):
    """RandomizedSearchCV best_params_ can contain numpy scalars; make them JSON-safe."""
    return value.item() if hasattr(value, "item") else value


def run(X, y, models_to_tune: list[str], baseline_scores: dict, cv_folds: int = 3,
        scoring: str = "roc_auc", n_iter: int = 8,
        output_dir: str = "outputs/pass_1") -> dict:
    """
    Runs RandomizedSearchCV for each model in models_to_tune that has a
    defined parameter grid. Compares against baseline_scores (model name ->
    Agent 1's untuned mean_test_score) to decide whether tuning helped.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    for name in models_to_tune:
        if name not in PARAM_DISTRIBUTIONS or name not in MODEL_REGISTRY:
            results.append({
                "model": name, "status": "skipped",
                "reason": "no tuning grid defined for this model",
            })
            print(f"  [skip] {name}: no tuning grid defined")
            continue

        base_estimator = build_model(name)
        # Avoid nested parallelism: if the estimator parallelizes internally
        # (n_jobs), force it to single-threaded so only the outer search
        # parallelizes. Prevents oversubscription/hangs.
        if "n_jobs" in base_estimator.get_params():
            base_estimator.set_params(n_jobs=1)

        search = RandomizedSearchCV(
            base_estimator, PARAM_DISTRIBUTIONS[name], n_iter=n_iter, cv=cv,
            scoring=scoring, random_state=42, n_jobs=-1, error_score="raise",
        )
        try:
            search.fit(X, y)
            baseline = baseline_scores.get(name)
            tuned_score = float(search.best_score_)
            improvement = (tuned_score - baseline) if baseline is not None else None
            best_params = {k: _to_jsonable(v) for k, v in search.best_params_.items()}

            results.append({
                "model": name,
                "status": "ok",
                "baseline_score": baseline,
                "tuned_score": tuned_score,
                "improvement": improvement,
                "beat_baseline": improvement is not None and improvement > 0,
                "best_params": best_params,
            })
            marker = "improved" if (improvement or 0) > 0 else "no improvement"
            print(f"  [ok] {name:20s} baseline={baseline:.4f} tuned={tuned_score:.4f} ({marker})")
        except Exception as e:
            results.append({"model": name, "status": f"error: {str(e)[:120]}"})
            print(f"  [fail] {name}: {str(e)[:100]}")

    with open(output_path / "agent5_tuning.json", "w") as f:
        json.dump(results, f, indent=2)

    return {"tuning_results": results}


if __name__ == "__main__":
    from orchestrator.data_utils import load_config

    config = load_config()
    pass_dir = f"outputs/pass_{config['pass_number']}"

    import pandas as pd
    with open(f"{pass_dir}/agent1_summary.json") as f:
        agent1_summary = json.load(f)
    leaderboard = pd.read_csv(f"{pass_dir}/leaderboard.csv")
    engineered = pd.read_csv(f"{pass_dir}/engineered_features.csv")

    target = config["dataset"]["target"]
    positive_class = config["dataset"]["positive_class"]
    X = engineered.drop(columns=[target])
    y = (engineered[target] == positive_class).astype(int)

    baseline_scores = dict(zip(leaderboard["model"], leaderboard["mean_test_score"]))
    top_models = agent1_summary["top_models"][:3]

    print(f"Running Agent 5 standalone: tuning {top_models}\n")
    result = run(
        X, y, top_models, baseline_scores,
        cv_folds=config.get("agent5_tuning", {}).get("cv_folds", 3),
        scoring=config["agent1_benchmarking"]["scoring"],
        n_iter=config.get("agent5_tuning", {}).get("n_iter", 8),
        output_dir=pass_dir,
    )
    print("\nResults:")
    print(json.dumps(result, indent=2))
