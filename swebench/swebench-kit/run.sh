#!/bin/bash
# swebench-kit 入口脚本。
#   ./run.sh --setup                          # 首次：用 uv 建 .venv（mini-swe-agent + swebench 评分器）
#   ./run.sh configs/qwen36.yaml              # 前台跑完整管线
#   ./run.sh configs/qwen36.yaml --detach     # 后台跑，ssh 断了也不死；进度看 runs/<tag>/kit.log
#   ./run.sh configs/qwen36.yaml --dry-run    # 只打印将要执行的命令，不实际运行
set -euo pipefail

# 106 rootless docker: 让 grade 阶段 docker SDK 找到 rootless socket (104 rootful 下自动 no-op)
if [ -z "${DOCKER_HOST:-}" ] && [ ! -S /var/run/docker.sock ] && [ -S "/run/user/$(id -u)/docker.sock" ]; then
  export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
fi

KIT="$(cd "$(dirname "$0")" && pwd)"

UV_BIN="$(command -v uv || true)"
[ -z "$UV_BIN" ] && [ -x /root/.local/bin/uv ] && UV_BIN=/root/.local/bin/uv

if [ "${1:-}" = "--setup" ]; then
  [ -z "$UV_BIN" ] && { echo "找不到 uv —— 安装：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  cd "$KIT" && "$UV_BIN" sync
  echo "OK: $KIT/.venv 就绪（$("$KIT/.venv/bin/python" -V)）"
  exit 0
fi

# 解释器优先级：kit 自己的 uv venv > 60017 上现成的 mini-venv（零安装兜底）
if [ -x "$KIT/.venv/bin/python" ]; then
  PY="$KIT/.venv/bin/python"
elif [ -x /xttest2/aiden/harness/mini-venv/bin/python ]; then
  PY=/xttest2/aiden/harness/mini-venv/bin/python
else
  echo "没有可用环境 —— 先执行：./run.sh --setup"; exit 1
fi

CFG="${1:?用法: run.sh CONFIG.yaml [--detach] [--dry-run] [--stages ...]}"
shift

# --detach 可以出现在任意位置；其余参数原样传给 kit.py
DETACH=0; ARGS=()
for a in "$@"; do
  [ "$a" = "--detach" ] && DETACH=1 || ARGS+=("$a")
done

if [ "$DETACH" = "1" ]; then
  mkdir -p "$KIT/runs"
  LOG="${KIT_DETACH_DIR:-$KIT/runs}/detach_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$LOG")"
  setsid "$PY" "$KIT/kit.py" "$CFG" ${ARGS[@]+"${ARGS[@]}"} >"$LOG" 2>&1 </dev/null &
  echo "已后台运行 pid=$! —— 进度：tail -f $LOG（以及 runs/<tag>/kit.log）"
else
  exec "$PY" "$KIT/kit.py" "$CFG" ${ARGS[@]+"${ARGS[@]}"}
fi
