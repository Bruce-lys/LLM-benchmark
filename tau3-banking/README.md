# τ³-Banking reproduction

This directory contains only the evaluation manifests and the files needed to
recreate the environment. It intentionally excludes the upstream source tree,
benchmark dataset, model logs, and evaluation results.

The run is pinned to
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench)
v1.0.1 (`fc0055dc4e0a316c3f83133267fbd6faaa770992`). The upstream project still
uses the `tau2` CLI for the τ³-Banking benchmark.

## Files

- `configs/macaron_v1_venti_l2_smoke.yaml`: one-task smoke test.
- `configs/macaron_v1_venti_l2_full.yaml`: 97-task Pass@1 evaluation.
- `scripts/setup.sh`: clones the pinned upstream release and installs the
  knowledge-domain dependencies.
- `scripts/run.sh`: loads a YAML manifest and launches the evaluation.

## Reproduce

Prerequisites: Linux, Python 3.12, Git, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd tau3-banking
cp .env.example .env
# Edit .env and insert the two API keys. Never commit .env.

./scripts/setup.sh
./scripts/run.sh configs/macaron_v1_venti_l2_smoke.yaml
./scripts/run.sh configs/macaron_v1_venti_l2_full.yaml
```

The configuration uses offline BM25 retrieval because the test server does not
allow the network namespace required by `bubblewrap`. Upstream benchmark data
is fetched only into the ignored `.tau2-bench/` directory during setup. Results
are written under `.tau2-bench/data/simulations/` and remain untracked.

The reference run used Macaron with `temperature=1.0` and `top_p=0.95`, and
GPT-5.2 with `reasoning_effort=low` as the user simulator.

