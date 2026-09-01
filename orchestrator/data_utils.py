"""Shared helpers for loading config and raw data across agents."""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "config.yaml") -> dict:
    path = PROJECT_ROOT / config_path
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    data_path = PROJECT_ROOT / config["dataset"]["path"]
    return pd.read_csv(data_path)
