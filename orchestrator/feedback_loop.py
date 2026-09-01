"""
Feedback loop.

This is the piece that makes the pipeline recursive rather than a one-shot
run. Before pass N starts, we check whether pass N-1 produced a
next_pass_config.json (written by Agent 4). If so, that config's
model_families / transforms override the base config.yaml for this pass.

Pass 1 has no prior pass, so it always runs on the base config.
"""

import json
from pathlib import Path


def load_effective_config(pass_number: int, base_config: dict) -> dict:
    """
    Returns the config that should actually be used for `pass_number`:
    - Pass 1: base_config as-is.
    - Pass N>1: base_config, but with agent1/agent2 settings overridden by
      the previous pass's next_pass_config.json if it exists.
    """
    config = json.loads(json.dumps(base_config))  # deep copy

    if pass_number == 1:
        return config

    prior_config_path = Path(f"outputs/pass_{pass_number - 1}/next_pass_config.json")
    if not prior_config_path.exists():
        print(f"  [feedback_loop] No next_pass_config.json found from pass {pass_number - 1}, "
              f"using base config.yaml as-is.")
        return config

    with open(prior_config_path) as f:
        carried = json.load(f)

    config["agent1_benchmarking"]["model_families"] = carried["agent1_benchmarking"]["model_families"]
    config["agent1_benchmarking"]["tuned_params"] = carried["agent1_benchmarking"].get("tuned_params", {})
    config["agent2_feature_engineering"]["transforms"] = carried["agent2_feature_engineering"]["transforms"]

    print(f"  [feedback_loop] Carried forward from pass {pass_number - 1}:")
    for reason in carried.get("carried_forward_reasoning", []):
        print(f"    - {reason}")

    return config
