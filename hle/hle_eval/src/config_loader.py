import json
import os
from copy import deepcopy


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)  # hle_eval/
DATA_DIR = os.path.join(ROOT_DIR, "data")
SUBSET_DIR = os.path.join(DATA_DIR, "subsets")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "predictions")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

DEFAULTS = {
    "dataset": "cais/hle",
    "subset_file": "data/subsets/hle_subset_10pct_category_difficulty_all.json",
    "parallel": 1,
    "realtime_judge": True,
    "text_only": False,
    "max_samples": None,
    "defaults": {
        "num_workers": 1,
        "temperature": 0.85,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.1,
        "repetition_penalty": 1.0,
        "timeout": 600.0,
        "question_timeout": 1800.0,
        "max_retries": 1,
        "max_completion_tokens": None,
        "token_param": "max_completion_tokens",
        "api_key_env": "OPENAI_API_KEY",
    },
    "judge": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "timeout": 180.0,
        "max_retries": 1,
    },
}


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if data is None:
        raise ValueError(f"Config is empty: {path}")
    return data


def _normalize_legacy_json(data):
    """Support old sglang_models.json list-of-models format."""
    if isinstance(data, list):
        return {"models": data}
    if isinstance(data, dict) and "models" in data:
        return data
    raise ValueError("Config must be a list of models or an object with a 'models' list.")


def resolve_path(path, config_dir=None, kind=None):
    """Resolve a path relative to config/root/subset/output dirs."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path

    candidates = []
    if config_dir:
        candidates.append(os.path.join(config_dir, path))
    candidates.append(os.path.join(ROOT_DIR, path))
    if kind == "subset":
        candidates.append(os.path.join(SUBSET_DIR, path))
        candidates.append(os.path.join(SUBSET_DIR, os.path.basename(path)))
    if kind == "output":
        candidates.append(os.path.join(OUTPUT_DIR, path))
        candidates.append(os.path.join(OUTPUT_DIR, os.path.basename(path)))
    candidates.append(path)

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Preferred destination when file does not exist yet.
    if kind == "output":
        if os.path.isabs(path) or path.startswith(("outputs/", "./outputs/")):
            return os.path.abspath(os.path.join(ROOT_DIR, path)) if not os.path.isabs(path) else path
        return os.path.abspath(os.path.join(OUTPUT_DIR, os.path.basename(path)))
    if kind == "subset":
        return os.path.abspath(os.path.join(SUBSET_DIR, os.path.basename(path)))
    if config_dir:
        return os.path.abspath(os.path.join(config_dir, path))
    return os.path.abspath(os.path.join(ROOT_DIR, path))


def safe_output_name(model_name):
    safe_name = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(OUTPUT_DIR, f"hle_{safe_name}.json")


def load_config(path):
    path = os.path.abspath(path)
    raw = _normalize_legacy_json(_load_raw(path))
    config = _deep_merge(DEFAULTS, raw)
    config["_config_path"] = path
    config["_config_dir"] = os.path.dirname(path)

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Config must contain a non-empty 'models' list: {path}")

    return config


def resolve_subset_file(config, cli_subset_file=None):
    if cli_subset_file:
        return resolve_path(cli_subset_file, config.get("_config_dir"), kind="subset")
    subset = config.get("subset_file")
    if not subset:
        return None
    return resolve_path(subset, config.get("_config_dir"), kind="subset")


def materialize_models(config, subset_file=None, max_samples=None):
    defaults = config.get("defaults", {})
    judge = config.get("judge", {})
    models = []

    for model in config["models"]:
        merged = deepcopy(defaults)
        merged.update({k: v for k, v in model.items() if v is not None})

        if "name" not in merged or "base_url" not in merged:
            raise ValueError("Each model requires 'name' and 'base_url'.")

        if not merged.get("output"):
            merged["output"] = safe_output_name(merged["name"])
        else:
            merged["output"] = resolve_path(merged["output"], ROOT_DIR, kind="output")

        if "realtime_judge" not in model:
            merged["realtime_judge"] = bool(config.get("realtime_judge", True))
        if "text_only" not in model:
            merged["text_only"] = bool(config.get("text_only", False))

        if subset_file is not None:
            merged["subset_file"] = subset_file
        elif "subset_file" in merged and merged["subset_file"]:
            merged["subset_file"] = resolve_path(
                merged["subset_file"], config.get("_config_dir"), kind="subset"
            )

        if max_samples is not None:
            merged["max_samples"] = max_samples
            merged.pop("subset_file", None)

        if merged.get("realtime_judge"):
            merged.setdefault("judge_model", judge.get("model"))
            merged.setdefault("judge_base_url", judge.get("base_url"))
            merged.setdefault("judge_api_key_env", judge.get("api_key_env"))
            merged.setdefault("judge_timeout", judge.get("timeout", 180.0))
            merged.setdefault("judge_max_retries", judge.get("max_retries", 1))

        models.append(merged)

    return models
