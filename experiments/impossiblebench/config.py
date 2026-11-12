"""Load and validate the shared ImpossibleBench experiment configuration."""

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.json")
EXPECTED_CONDITIONS = {"base", "pre_conceal_train", "post_conceal_train"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the config and fail early when a model pin is missing."""
    config = json.loads(path.read_text())

    for artifact in (config["base_model"], config["tokenizer"]):
        _validate_artifact(artifact)

    if not config["modal"].get("gpu"):
        raise ValueError("Missing Modal GPU type")

    conditions = config["conditions"]
    if set(conditions) != EXPECTED_CONDITIONS:
        raise ValueError(f"Expected conditions: {sorted(EXPECTED_CONDITIONS)}")

    for name, condition in conditions.items():
        for field in ("label", "behavior_train_method", "conceal_train_method"):
            if not condition.get(field):
                raise ValueError(f"{name} is missing {field}")

        adapter = condition["adapter"]
        if name == "base" and adapter is not None:
            raise ValueError("The base condition cannot have an adapter")
        if name != "base":
            _validate_artifact(adapter)
            if not isinstance(adapter["lora_rank"], int):
                raise ValueError(f"{name} lora_rank must be an integer")

    return config


def _validate_artifact(artifact: dict) -> None:
    for field in ("repo_id", "revision"):
        if not artifact.get(field):
            raise ValueError(f"Missing artifact field: {field}")
    _validate_revision(artifact["revision"])


def _validate_revision(revision: str) -> None:
    if len(revision) != 40:
        raise ValueError(f"Revision must be a 40-character commit: {revision}")


if __name__ == "__main__":
    print(json.dumps(load_config(), indent=2))
