#!/usr/bin/env python3
"""Launch a pinned tau2 run from a small, reviewable YAML manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


def expand_environment(value: Any) -> Any:
    """Expand environment variables recursively and reject missing values."""
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable in: {value}")
        if expanded.startswith("replace_with_"):
            raise ValueError(f"Placeholder value has not been replaced: {value}")
        return expanded
    return value


def add_option(command: list[str], name: str, value: Any) -> None:
    """Append a scalar CLI option when it has a value."""
    if value is not None:
        command.extend([f"--{name.replace('_', '-')}", str(value)])


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1]).resolve()
    config = expand_environment(yaml.safe_load(config_path.read_text()))
    benchmark = config["benchmark"]
    agent = config["agent"]
    user = config["user_simulator"]
    run = config["run"]

    tau2_dir = Path(os.environ["TAU2_DIR"]).resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(tau2_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = benchmark["upstream_commit"]
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"tau2-bench commit mismatch: {actual_commit}; expected {expected_commit}"
        )

    command = ["uv", "run", "--project", str(tau2_dir), "tau2", "run"]
    add_option(command, "domain", benchmark["domain"])
    add_option(command, "retrieval_config", benchmark["retrieval_config"])
    add_option(command, "agent_llm", agent["llm"])
    add_option(command, "agent_llm_args", json.dumps(agent["llm_args"]))
    add_option(command, "user_llm", user["llm"])
    add_option(command, "user_llm_args", json.dumps(user["llm_args"]))

    task_ids = run.get("task_ids")
    if task_ids:
        command.append("--task-ids")
        command.extend(task_ids)

    for option in (
        "num_trials",
        "max_concurrency",
        "max_steps",
        "timeout",
        "max_retries",
        "retry_delay",
        "seed",
        "save_to",
        "llm_log_mode",
    ):
        add_option(command, option, run.get(option))

    for flag in ("auto_resume", "verbose_logs"):
        if run.get(flag):
            command.append(f"--{flag.replace('_', '-')}")

    print(
        f"Running {config_path.name} with pinned tau2-bench {expected_commit[:8]} "
        "(API keys hidden)."
    )
    return subprocess.run(command, cwd=tau2_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

