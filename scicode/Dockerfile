# SciCode eval environment (uv-based; portable)
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY eval ./eval
COPY run_aa_eval.sh run_smoke.sh .env.example ./

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e . && \
    chmod +x /app/run_aa_eval.sh /app/run_smoke.sh

# Mount host network / pass OPENAI_BASE_URL to reach vLLM:
#   docker run --rm --network host \
#     -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
#     -e OPENAI_API_KEY=EMPTY \
#     -v $PWD/eval/data/test_data.h5:/app/eval/data/test_data.h5:ro \
#     scicode-eval ./run_aa_eval.sh

ENV PATH="/app/.venv/bin:$PATH" \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=EMPTY

ENTRYPOINT ["./run_aa_eval.sh"]
