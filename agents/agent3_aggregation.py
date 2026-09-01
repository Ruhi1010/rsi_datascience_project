"""
Agent 3: results aggregation agent.

Job: merge Agent 1's leaderboard with Agent 2's feature engineering log and
synthesize what happened — not just list the numbers, but explain what they
mean together. This is the layer where the system starts reasoning about
its own results instead of just producing them.

Output: aggregation_summary.json consumed by Agent 4.
"""

import json
from pathlib import Path

import pandas as pd

# Models that are scale/distance sensitive vs. scale-invariant.
# Used to explain *why* certain models moved after feature engineering.
SCALE_SENSITIVE_MODELS = {"logistic_regression", "knn", "svm_linear", "naive_bayes"}
SCALE_INVARIANT_MODELS = {
    "decision_tree", "random_forest", "extra_trees",
    "gradient_boosting", "adaboost", "xgboost", "lightgbm",
}


def run(agent1_summary: dict, agent2_summary: dict, leaderboard: pd.DataFrame,
        prior_leaderboard: pd.DataFrame | None = None,
        agent5_summary: dict | None = None,
        output_dir: str = "outputs/pass_1") -> dict:
    """
    Combines Agent 1 + Agent 2 (+ optionally Agent 5 tuning) outputs into a
    synthesis. If a prior pass's leaderboard is provided, also compares
    across passes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ok = leaderboard[leaderboard["status"] == "ok"].copy()

    patterns = []

    # Pattern 1: did feature engineering help scale-sensitive models more
    # than scale-invariant ones? Only meaningful if we have a prior pass
    # without scaling to compare against, but we can still flag which
    # category benefited based on the applied transforms.
    applied = set(agent2_summary.get("transforms_applied", []))
    if "numeric_scaling" in applied:
        sensitive_present = ok[ok["model"].isin(SCALE_SENSITIVE_MODELS)]
        invariant_present = ok[ok["model"].isin(SCALE_INVARIANT_MODELS)]
        if len(sensitive_present) and len(invariant_present):
            patterns.append(
                f"Numeric scaling was applied. Scale-sensitive models "
                f"(mean {agent1_summary['scoring_metric']}={sensitive_present['mean_test_score'].mean():.4f}) "
                f"benefit most from this transform; tree-based models "
                f"(mean {invariant_present['mean_test_score'].mean():.4f}) are largely unaffected by scaling."
            )

    # Pattern 2: overfitting models flagged by Agent 1
    if agent1_summary.get("overfitting_models"):
        patterns.append(
            f"Models showing overfitting (large train/test gap): "
            f"{', '.join(agent1_summary['overfitting_models'])}. "
            f"These are candidates for regularization or depth limits next pass."
        )

    # Pattern 3: unstable models
    if agent1_summary.get("unstable_models"):
        patterns.append(
            f"Models with high variance across folds: "
            f"{', '.join(agent1_summary['unstable_models'])}. "
            f"Results for these should be treated cautiously."
        )

    # Pattern 4: feature count vs. performance
    n_added = agent2_summary.get("n_features_added", 0)
    if n_added > 0:
        patterns.append(
            f"Feature engineering added {n_added} new columns "
            f"({agent2_summary['shape_before'][1]} -> {agent2_summary['shape_after'][1]}). "
            f"Best model after engineering: {agent1_summary['best_model']} "
            f"at {agent1_summary['best_score']:.4f}."
        )

    # Pattern 5: cross-pass comparison, if a prior pass exists
    pass_over_pass = None
    if prior_leaderboard is not None:
        prior_ok = prior_leaderboard[prior_leaderboard["status"] == "ok"]
        if len(prior_ok):
            prior_best = prior_ok.iloc[0]
            current_best = ok.iloc[0] if len(ok) else None
            if current_best is not None:
                delta = current_best["mean_test_score"] - prior_best["mean_test_score"]
                direction = "improved" if delta > 0 else "declined" if delta < 0 else "unchanged"
                pass_over_pass = {
                    "prior_best_model": prior_best["model"],
                    "prior_best_score": float(prior_best["mean_test_score"]),
                    "current_best_model": current_best["model"],
                    "current_best_score": float(current_best["mean_test_score"]),
                    "delta": float(delta),
                    "direction": direction,
                }
                patterns.append(
                    f"Compared to the previous pass, the best score {direction} "
                    f"by {abs(delta):.4f} ({prior_best['model']}={prior_best['mean_test_score']:.4f} "
                    f"-> {current_best['model']}={current_best['mean_test_score']:.4f})."
                )

    # Pattern 6: tuning results, if Agent 5 ran this pass
    tuning_improved = []
    tuning_no_gain = []
    if agent5_summary:
        for r in agent5_summary.get("tuning_results", []):
            if r.get("status") != "ok":
                continue
            if r.get("beat_baseline"):
                tuning_improved.append(
                    f"{r['model']} ({r['baseline_score']:.4f} -> {r['tuned_score']:.4f}, "
                    f"+{r['improvement']:.4f})"
                )
            else:
                tuning_no_gain.append(r["model"])
        if tuning_improved:
            patterns.append(
                f"Hyperparameter tuning improved: {'; '.join(tuning_improved)}. "
                f"These tuned settings will be carried into the next pass."
            )
        if tuning_no_gain:
            patterns.append(
                f"Hyperparameter tuning did not beat the untuned baseline for: "
                f"{', '.join(tuning_no_gain)}. Default hyperparameters remain in use."
            )

    # Recommendations for the next pass / for Agent 4 to surface
    recommendations = []
    if agent1_summary.get("overfitting_models"):
        recommendations.append(
            f"Add regularization or reduce depth for: {', '.join(agent1_summary['overfitting_models'])}"
        )
    if not agent5_summary or not agent5_summary.get("tuning_results"):
        if agent1_summary.get("top_models"):
            recommendations.append(
                f"Deepen hyperparameter tuning for top performers: {', '.join(agent1_summary['top_models'][:3])}"
            )
    if "interaction_terms" not in applied:
        recommendations.append("Try adding interaction_terms transform if not already applied")
    if "aggregated_features" not in applied:
        recommendations.append("Try adding aggregated_features transform if not already applied")

    summary = {
        "pass_scoring_metric": agent1_summary["scoring_metric"],
        "best_model": agent1_summary.get("best_model"),
        "best_score": agent1_summary.get("best_score"),
        "n_models_evaluated": agent1_summary.get("n_models_succeeded"),
        "n_features_final": agent2_summary["shape_after"][1],
        "patterns": patterns,
        "recommendations": recommendations,
        "pass_over_pass_comparison": pass_over_pass,
        "tuning_improved_models": tuning_improved,
        "tuning_no_gain_models": tuning_no_gain,
    }

    with open(output_path / "agent3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from orchestrator.data_utils import load_config

    config = load_config()
    pass_dir = f"outputs/pass_{config['pass_number']}"

    with open(f"{pass_dir}/agent1_summary.json") as f:
        agent1_summary = json.load(f)
    with open(f"{pass_dir}/agent2_summary.json") as f:
        agent2_summary = json.load(f)
    leaderboard = pd.read_csv(f"{pass_dir}/leaderboard.csv")

    print("Running Agent 3 standalone (aggregating pass 1 results)\n")
    summary = run(agent1_summary, agent2_summary, leaderboard, output_dir=pass_dir)

    print("Patterns found:")
    for p in summary["patterns"]:
        print(f"  - {p}")
    print("\nRecommendations for next pass:")
    for r in summary["recommendations"]:
        print(f"  - {r}")
