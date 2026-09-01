"""
Agent 4: reporting agent.

Job: turn Agent 1/2/3's outputs into a readable end-to-end report, and
produce a structured "next_pass_actions" block that the feedback loop
uses to configure Agent 1 and Agent 2 for the next pass.

Output: report.md (human-readable) + next_pass_config.json (machine-readable,
consumed by orchestrator/feedback_loop.py).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Simple rule table: if agent3 recommends tuning a model, which new
# candidate models (not yet tried) should agent1 add next pass.
RELATED_MODEL_SUGGESTIONS = {
    "lightgbm": [],       # already the strongest family, nothing further to add
    "xgboost": [],
    "gradient_boosting": ["hist_gradient_boosting"],
    "random_forest": [],
}


def _build_next_pass_config(agent1_summary: dict, agent2_summary: dict,
                             agent3_summary: dict, current_config: dict,
                             agent5_summary: dict | None = None) -> dict:
    """Derives the config for the next pass from this pass's findings."""
    next_model_families = list(current_config["agent1_benchmarking"]["model_families"])
    next_transforms = list(current_config["agent2_feature_engineering"]["transforms"])
    next_tuned_params = dict(current_config["agent1_benchmarking"].get("tuned_params", {}))

    # If agent3 flagged missing transforms, add them.
    for rec in agent3_summary.get("recommendations", []):
        if "interaction_terms" in rec and "interaction_terms" not in next_transforms:
            next_transforms.append("interaction_terms")
        if "aggregated_features" in rec and "aggregated_features" not in next_transforms:
            next_transforms.append("aggregated_features")
        if "domain_specific_transformations" in rec and "domain_specific_transformations" not in next_transforms:
            next_transforms.append("domain_specific_transformations")

    # If a top model has a known related family not yet tried, add it.
    for model in agent1_summary.get("top_models", []):
        for candidate in RELATED_MODEL_SUGGESTIONS.get(model, []):
            if candidate not in next_model_families:
                next_model_families.append(candidate)

    # Carry forward winning hyperparameters from Agent 5's tuning this pass.
    if agent5_summary:
        for r in agent5_summary.get("tuning_results", []):
            if r.get("status") == "ok" and r.get("beat_baseline"):
                next_tuned_params[r["model"]] = r["best_params"]

    return {
        "pass_number": current_config["pass_number"] + 1,
        "agent1_benchmarking": {
            **current_config["agent1_benchmarking"],
            "model_families": next_model_families,
            "tuned_params": next_tuned_params,
        },
        "agent2_feature_engineering": {
            **current_config["agent2_feature_engineering"],
            "transforms": next_transforms,
        },
        "carried_forward_reasoning": agent3_summary.get("recommendations", []),
    }


def _render_markdown(pass_number: int, agent1_summary: dict, agent2_summary: dict,
                      agent3_summary: dict, leaderboard: pd.DataFrame,
                      agent5_summary: dict | None = None) -> str:
    lines = []
    lines.append(f"# Pipeline report — pass {pass_number}")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n")

    lines.append("## Dataset summary")
    lines.append(f"- Final feature count: {agent2_summary['shape_after'][1]} "
                 f"(started at {agent2_summary['shape_before'][1]})")
    lines.append(f"- Transforms applied: {', '.join(agent2_summary['transforms_applied'])}\n")

    lines.append("## Modeling summary")
    lines.append(f"- Models evaluated: {agent1_summary['n_models_succeeded']}/{agent1_summary['n_models_attempted']}")
    lines.append(f"- Scoring metric: {agent1_summary['scoring_metric']}")
    lines.append(f"- Best model: **{agent1_summary['best_model']}** "
                 f"({agent1_summary['best_score']:.4f})\n")

    lines.append("## Leaderboard")
    lines.append("| Model | Score | Std | Overfit gap |")
    lines.append("|---|---|---|---|")
    ok = leaderboard[leaderboard["status"] == "ok"]
    for _, row in ok.iterrows():
        lines.append(f"| {row['model']} | {row['mean_test_score']:.4f} | "
                     f"{row['std_test_score']:.4f} | {row['overfit_gap']:.4f} |")
    lines.append("")

    lines.append("## Feature engineering summary")
    for line in agent2_summary["change_log"]:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## Best-performing approaches")
    for model in agent1_summary.get("top_models", []):
        lines.append(f"- {model}")
    lines.append("")

    lines.append("## Weaknesses and failure cases")
    if agent1_summary.get("unstable_models"):
        lines.append(f"- Unstable across folds: {', '.join(agent1_summary['unstable_models'])}")
    if agent1_summary.get("overfitting_models"):
        lines.append(f"- Overfitting (train/test gap): {', '.join(agent1_summary['overfitting_models'])}")
    if agent2_summary.get("transforms_skipped"):
        lines.append(f"- Transforms skipped (not in registry): {', '.join(agent2_summary['transforms_skipped'])}")
    lines.append("")

    lines.append("## Patterns observed")
    for p in agent3_summary.get("patterns", []):
        lines.append(f"- {p}")
    lines.append("")

    if agent5_summary and agent5_summary.get("tuning_results"):
        lines.append("## Hyperparameter tuning")
        for r in agent5_summary["tuning_results"]:
            if r.get("status") != "ok":
                lines.append(f"- {r['model']}: {r.get('reason', r.get('status'))}")
                continue
            verdict = "beat baseline" if r["beat_baseline"] else "did not beat baseline"
            lines.append(
                f"- **{r['model']}**: baseline {r['baseline_score']:.4f} -> "
                f"tuned {r['tuned_score']:.4f} ({verdict})"
            )
            if r["beat_baseline"]:
                lines.append(f"  - Winning params: `{r['best_params']}`")
        lines.append("")

    if agent3_summary.get("pass_over_pass_comparison"):
        cmp = agent3_summary["pass_over_pass_comparison"]
        lines.append("## Pass-over-pass comparison")
        lines.append(f"- Previous best: {cmp['prior_best_model']} ({cmp['prior_best_score']:.4f})")
        lines.append(f"- Current best: {cmp['current_best_model']} ({cmp['current_best_score']:.4f})")
        lines.append(f"- Change: {cmp['direction']} by {abs(cmp['delta']):.4f}\n")

    lines.append("## Recommendations for the next iteration")
    for r in agent3_summary.get("recommendations", []):
        lines.append(f"- {r}")

    return "\n".join(lines)


def run(pass_number: int, agent1_summary: dict, agent2_summary: dict,
        agent3_summary: dict, leaderboard: pd.DataFrame, current_config: dict,
        agent5_summary: dict | None = None,
        output_dir: str = "outputs/pass_1") -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_md = _render_markdown(pass_number, agent1_summary, agent2_summary,
                                  agent3_summary, leaderboard, agent5_summary)
    with open(output_path / "report.md", "w") as f:
        f.write(report_md)

    next_pass_config = _build_next_pass_config(
        agent1_summary, agent2_summary, agent3_summary, current_config, agent5_summary
    )
    with open(output_path / "next_pass_config.json", "w") as f:
        json.dump(next_pass_config, f, indent=2)

    return {
        "report_path": str(output_path / "report.md"),
        "next_pass_config_path": str(output_path / "next_pass_config.json"),
        "next_pass_config": next_pass_config,
    }


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
    with open(f"{pass_dir}/agent3_summary.json") as f:
        agent3_summary = json.load(f)
    leaderboard = pd.read_csv(f"{pass_dir}/leaderboard.csv")

    agent5_summary = None
    agent5_path = Path(f"{pass_dir}/agent5_tuning.json")
    if agent5_path.exists():
        with open(agent5_path) as f:
            agent5_summary = {"tuning_results": json.load(f)}

    print("Running Agent 4 standalone (reporting on pass 1)\n")
    result = run(
        config["pass_number"], agent1_summary, agent2_summary,
        agent3_summary, leaderboard, config, agent5_summary=agent5_summary,
        output_dir=pass_dir,
    )
    print(f"Report written to: {result['report_path']}")
    print(f"Next pass config written to: {result['next_pass_config_path']}")
    print("\nNext pass will use:")
    print(f"  model_families: {result['next_pass_config']['agent1_benchmarking']['model_families']}")
    print(f"  transforms: {result['next_pass_config']['agent2_feature_engineering']['transforms']}")
