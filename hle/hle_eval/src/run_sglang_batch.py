import argparse
import os
import subprocess
import sys

from config_loader import SRC_DIR, load_config, materialize_models, resolve_subset_file


def optional_value(model_config, key, default):
    value = model_config.get(key, default)
    return default if value is None else value


def filter_models(models, args):
    if args.batch is not None:
        models = [model for model in models if str(model.get("batch")) == str(args.batch)]

    if args.only:
        selected_names = {name.strip() for name in args.only.split(",") if name.strip()}
        models = [model for model in models if model.get("name") in selected_names]

    if not models:
        raise ValueError("No models selected. Check --batch, --only, or the config file.")

    return models


def build_command(args, model_config, progress_position=0):
    model_name = model_config["name"]
    command = [
        sys.executable,
        os.path.join(SRC_DIR, "run_model_predictions.py"),
        "--dataset",
        optional_value(model_config, "dataset", args.dataset),
        "--model",
        model_name,
        "--base_url",
        model_config["base_url"],
        "--api_key_env",
        optional_value(model_config, "api_key_env", args.api_key_env),
        "--output",
        model_config["output"],
        "--num_workers",
        str(optional_value(model_config, "num_workers", args.num_workers)),
        "--temperature",
        str(optional_value(model_config, "temperature", args.temperature)),
        "--timeout",
        str(optional_value(model_config, "timeout", args.timeout)),
        "--max_retries",
        str(optional_value(model_config, "max_retries", args.max_retries)),
    ]

    question_timeout = optional_value(model_config, "question_timeout", args.question_timeout)
    if question_timeout is not None:
        command.extend(["--question_timeout", str(question_timeout)])

    top_p = optional_value(model_config, "top_p", args.top_p)
    if top_p is not None:
        command.extend(["--top_p", str(top_p)])

    top_k = optional_value(model_config, "top_k", args.top_k)
    if top_k is not None:
        command.extend(["--top_k", str(top_k)])

    min_p = optional_value(model_config, "min_p", args.min_p)
    if min_p is not None:
        command.extend(["--min_p", str(min_p)])

    presence_penalty = optional_value(model_config, "presence_penalty", args.presence_penalty)
    if presence_penalty is not None:
        command.extend(["--presence_penalty", str(presence_penalty)])

    repetition_penalty = optional_value(model_config, "repetition_penalty", args.repetition_penalty)
    if repetition_penalty is not None:
        command.extend(["--repetition_penalty", str(repetition_penalty)])

    max_completion_tokens = optional_value(
        model_config,
        "max_completion_tokens",
        args.max_completion_tokens,
    )
    if max_completion_tokens is not None:
        command.extend(["--max_completion_tokens", str(max_completion_tokens)])
        command.extend(["--token_param", optional_value(model_config, "token_param", args.token_param)])

    subset_file = optional_value(model_config, "subset_file", args.subset_file)
    if subset_file is not None:
        command.extend(["--subset_file", subset_file])
    else:
        max_samples = optional_value(model_config, "max_samples", args.max_samples)
        if max_samples is not None:
            command.extend(["--max_samples", str(max_samples)])

    if optional_value(model_config, "text_only", args.text_only):
        command.append("--text_only")

    command.extend(["--progress_position", str(progress_position)])

    if optional_value(model_config, "realtime_judge", args.realtime_judge):
        command.append("--realtime_judge")
        command.extend(["--judge_model", optional_value(model_config, "judge_model", args.judge_model)])
        command.extend(["--judge_base_url", optional_value(model_config, "judge_base_url", args.judge_base_url)])
        command.extend(["--judge_api_key_env", optional_value(model_config, "judge_api_key_env", args.judge_api_key_env)])
        command.extend(["--judge_timeout", str(optional_value(model_config, "judge_timeout", args.judge_timeout))])
        command.extend(["--judge_max_retries", str(optional_value(model_config, "judge_max_retries", args.judge_max_retries))])

    return command


def validate_model_config(model_config):
    missing = [key for key in ("name", "base_url") if not model_config.get(key)]
    if missing:
        raise ValueError(f"Model config is missing required field(s): {', '.join(missing)}")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_commands(commands, dry_run, parallel):
    if dry_run:
        return

    for start in range(0, len(commands), parallel):
        chunk = commands[start:start + parallel]
        processes = [subprocess.Popen(command) for command in chunk]
        failures = []
        for command, process in zip(chunk, processes):
            return_code = process.wait()
            if return_code != 0:
                failures.append((" ".join(command), return_code))

        if failures:
            for command, return_code in failures:
                print(f"Command failed with exit code {return_code}: {command}", file=sys.stderr)
            raise subprocess.CalledProcessError(failures[0][1], failures[0][0])


def main(args):
    if args.parallel is not None and args.parallel < 1:
        raise ValueError("--parallel must be 1 or greater.")

    config = load_config(args.config)
    defaults = config["defaults"]
    judge = config["judge"]

    if args.dataset is None:
        args.dataset = config["dataset"]
    if args.api_key_env is None:
        args.api_key_env = defaults.get("api_key_env", "OPENAI_API_KEY")
    if args.num_workers is None:
        args.num_workers = defaults.get("num_workers", 1)
    if args.temperature is None:
        args.temperature = defaults.get("temperature", 0.0)
    if args.top_p is None:
        args.top_p = defaults.get("top_p")
    if args.top_k is None:
        args.top_k = defaults.get("top_k")
    if args.min_p is None:
        args.min_p = defaults.get("min_p")
    if args.presence_penalty is None:
        args.presence_penalty = defaults.get("presence_penalty")
    if args.repetition_penalty is None:
        args.repetition_penalty = defaults.get("repetition_penalty")
    if args.max_completion_tokens is None:
        args.max_completion_tokens = defaults.get("max_completion_tokens")
    if args.token_param is None:
        args.token_param = defaults.get("token_param", "max_completion_tokens")
    if args.timeout is None:
        args.timeout = defaults.get("timeout", 600.0)
    if args.question_timeout is None:
        args.question_timeout = defaults.get("question_timeout", 1800.0)
    if args.max_retries is None:
        args.max_retries = defaults.get("max_retries", 1)
    if args.parallel is None:
        # parallel lives at YAML top-level (merged into config), not under defaults.*
        args.parallel = int(config.get("parallel", defaults.get("parallel", 1)))
    if not args.text_only:
        args.text_only = bool(config.get("text_only", False))
    if not args.realtime_judge:
        args.realtime_judge = bool(config.get("realtime_judge", False))
    if args.judge_model is None:
        args.judge_model = config.get("judge_model") or judge.get("model")
    if args.judge_base_url is None:
        args.judge_base_url = config.get("judge_base_url") or judge.get("base_url")
    if args.judge_api_key_env is None:
        args.judge_api_key_env = config.get("judge_api_key_env") or judge.get("api_key_env")
    if args.judge_timeout is None:
        args.judge_timeout = config.get("judge_timeout") or judge.get("timeout", 180.0)
    if args.judge_max_retries is None:
        args.judge_max_retries = config.get("judge_max_retries") or judge.get("max_retries", 1)

    subset_file = resolve_subset_file(config, args.subset_file)
    args.subset_file = subset_file

    models = filter_models(
        materialize_models(config, subset_file=subset_file, max_samples=args.max_samples),
        args,
    )

    commands = []
    for index, model_config in enumerate(models, start=1):
        validate_model_config(model_config)
        ensure_parent_dir(model_config["output"])
        command = build_command(args, model_config, progress_position=index - 1)
        print(f"\n[{index}/{len(models)}] Running {model_config['name']}")
        print(" ".join(command))
        commands.append(command)

    run_commands(commands, args.dry_run, args.parallel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run HLE predictions from a YAML/JSON config.")
    parser.add_argument("--config", type=str, required=True, help="YAML/JSON config with model endpoints.")
    parser.add_argument("--dataset", type=str, default=None, help="Override HLE dataset.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Override API key environment variable.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override prediction concurrency.")
    parser.add_argument("--max_completion_tokens", type=int, default=None, help="Override max completion tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Override top-p sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Override top-k sampling.")
    parser.add_argument("--min_p", type=float, default=None, help="Override min-p sampling.")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Override presence penalty.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--token_param", choices=["max_completion_tokens", "max_tokens"], default=None, help="Override token limit parameter.")
    parser.add_argument("--timeout", type=float, default=None, help="Override OpenAI/httpx idle timeout in seconds.")
    parser.add_argument(
        "--question_timeout",
        type=float,
        default=None,
        help="Override hard wall-clock timeout per question in seconds.",
    )
    parser.add_argument("--max_retries", type=int, default=None, help="Override OpenAI client max retries.")
    parser.add_argument("--realtime_judge", action="store_true", help="Enable realtime judge for all models.")
    parser.add_argument("--judge_model", type=str, default=None, help="Realtime judge model.")
    parser.add_argument("--judge_base_url", type=str, default=None, help="Realtime judge base URL.")
    parser.add_argument("--judge_api_key_env", type=str, default=None, help="Realtime judge API key env var.")
    parser.add_argument("--judge_timeout", type=float, default=None, help="Realtime judge timeout.")
    parser.add_argument("--judge_max_retries", type=int, default=None, help="Realtime judge max retries.")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max samples for all models (smoke).")
    parser.add_argument("--subset_file", type=str, default=None, help="Fixed subset JSON shared by all models.")
    parser.add_argument("--text_only", action="store_true", help="Skip image questions for all models.")
    parser.add_argument("--batch", type=str, default=None, help="Run only models with this batch id.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated model names to run.")
    parser.add_argument("--parallel", type=int, default=None, help="Number of model endpoints to evaluate concurrently.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    main(args)
