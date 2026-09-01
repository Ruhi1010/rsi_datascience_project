"""
Orchestrator: runs one full pass through all four agents in sequence.

    Agent 1 (benchmark raw/prior features)
        -> Agent 2 (engineer features)
        -> Agent 1 again on engineered features   [see note below]
        -> Agent 3 (aggregate)
        -> Agent 4 (report + next_pass_config)

Note on ordering: Agent 2 needs the raw dataframe (to encode/impute raw
columns), and Agent 1 needs engineered features to benchmark against. So
in practice Agent 2 runs first, then Agent 1 runs on its output. Agent 1's
"job" (which models work, which are unstable) still comes first
conceptually in the report — the execution order and the reporting order
don't have to match.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents import agent1_benchmarking, agent2_feature_eng, agent3_aggregation, agent4_reporting, agent5_tuning
from orchestrator.data_utils import load_config, load_raw_data
from orchestrator.feedback_loop import load_effective_config


def run_pass(pass_number: int, base_config: dict) -> dict:
    output_dir = f"outputs/pass_{pass_number}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = load_effective_config(pass_number, base_config)
    config["pass_number"] = pass_number

    df = load_raw_data(config)
    target = config["dataset"]["target"]
    positive_class = config["dataset"]["positive_class"]

    print(f"\n{'='*60}\nPASS {pass_number}\n{'='*60}")

    # --- Agent 2: feature engineering ---
    print("\n[Agent 2] Feature engineering...")
    agent2_summary = agent2_feature_eng.run(
        df, target,
        transforms=config["agent2_feature_engineering"]["transforms"],
        output_dir=output_dir,
    )
    print(f"  Applied: {agent2_summary['transforms_applied']}")
    print(f"  Shape: {agent2_summary['shape_before']} -> {agent2_summary['shape_after']}")

    engineered = pd.read_csv(agent2_summary["output_path"])
    X = engineered.drop(columns=[target])
    y = (engineered[target] == positive_class).astype(int)

    # --- Agent 1: model benchmarking ---
    print("\n[Agent 1] Model benchmarking...")
    tuned_params = config["agent1_benchmarking"].get("tuned_params", {})
    agent1_summary = agent1_benchmarking.run(
        X, y,
        model_families=config["agent1_benchmarking"]["model_families"],
        cv_folds=config["agent1_benchmarking"]["cv_folds"],
        scoring=config["agent1_benchmarking"]["scoring"],
        output_dir=output_dir,
        tuned_params=tuned_params,
    )
    print(f"  Best: {agent1_summary['best_model']} ({agent1_summary['best_score']:.4f})")

    leaderboard = pd.read_csv(f"{output_dir}/leaderboard.csv")

    # --- Agent 5: hyperparameter tuning ---
    print("\n[Agent 5] Hyperparameter tuning...")
    n_top = config.get("agent5_tuning", {}).get("n_top_models", 3)
    n_iter = config.get("agent5_tuning", {}).get("n_iter", 20)
    models_to_tune = [m for m in agent1_summary.get("top_models", [])[:n_top]
                       if m not in tuned_params]
    baseline_scores = dict(zip(leaderboard["model"], leaderboard["mean_test_score"]))

    if models_to_tune:
        tuning_cv_folds = config.get("agent5_tuning", {}).get("cv_folds", 3)
        agent5_summary = agent5_tuning.run(
            X, y, models_to_tune, baseline_scores,
            cv_folds=tuning_cv_folds,
            scoring=config["agent1_benchmarking"]["scoring"],
            n_iter=n_iter, output_dir=output_dir,
        )
    else:
        print("  Nothing to tune (top models already tuned in a prior pass).")
        agent5_summary = {"tuning_results": []}

    # --- Agent 3: results aggregation ---
    print("\n[Agent 3] Aggregating results...")
    prior_leaderboard = None
    prior_dir = Path(f"outputs/pass_{pass_number - 1}")
    if pass_number > 1 and (prior_dir / "leaderboard.csv").exists():
        prior_leaderboard = pd.read_csv(prior_dir / "leaderboard.csv")

    agent3_summary = agent3_aggregation.run(
        agent1_summary, agent2_summary, leaderboard,
        prior_leaderboard=prior_leaderboard, agent5_summary=agent5_summary,
        output_dir=output_dir,
    )
    print(f"  {len(agent3_summary['patterns'])} patterns found")

    # --- Agent 4: reporting ---
    print("\n[Agent 4] Writing report + next-pass config...")
    agent4_result = agent4_reporting.run(
        pass_number, agent1_summary, agent2_summary, agent3_summary,
        leaderboard, config, agent5_summary=agent5_summary, output_dir=output_dir,
    )
    print(f"  Report: {agent4_result['report_path']}")

    return {
        "pass_number": pass_number,
        "best_model": agent1_summary["best_model"],
        "best_score": agent1_summary["best_score"],
        "n_features": agent2_summary["shape_after"][1],
        "n_models_evaluated": agent1_summary["n_models_succeeded"],
        "overfitting_models": len(agent1_summary.get("overfitting_models", [])),
        "unstable_models": len(agent1_summary.get("unstable_models", [])),
        "models_tuned": len([r for r in agent5_summary.get("tuning_results", []) if r.get("status") == "ok"]),
        "tuning_improvements": len(agent3_summary.get("tuning_improved_models", [])),
    }


def update_metrics_log(pass_result: dict, metrics_path: str = "metrics/pass_comparison.csv"):
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([pass_result])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["pass_number"] != pass_result["pass_number"]]
        combined = pd.concat([existing, row], ignore_index=True).sort_values("pass_number")
    else:
        combined = row
    combined.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="pass_number", type=int, required=True)
    args = parser.parse_args()

    base_config = load_config()
    result = run_pass(args.pass_number, base_config)
    update_metrics_log(result)

    print(f"\n{'='*60}")
    print(f"Pass {args.pass_number} complete: {result['best_model']} = {result['best_score']:.4f}")
    print(f"{'='*60}")
