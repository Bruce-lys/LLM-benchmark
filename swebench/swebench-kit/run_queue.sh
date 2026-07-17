#!/bin/bash
# swebench-kit 轮回跑：按顺序跑多个配置，每个跑完自动出结果 → 卸载模型 → 加载下一个。
#   ./run_queue.sh a.yaml b.yaml c.yaml         # 前台按序跑完所有配置
#   ./run_queue.sh --detach a.yaml b.yaml       # 整个队列后台跑（ssh 断了也不死）
#   ./run_queue.sh --list queue.txt             # 从文件读配置列表（每行一个路径，支持 # 注释）
#
# 原理：每个配置就是一次完整的 serve→gen→grade（kit.py 正常退出时 atexit 自动关闭
# 它自己拉起的 vLLM 进程组）。本脚本只负责四件事：
#   1. 排队依次执行；单个配置失败不中断队列，记录后继续下一个
#   2. 两个模型之间等待端口 + 显存释放，避免下一个 vLLM 绑不上端口
#   3. 兜底清理：kit 被 SIGKILL 等极端情况下 atexit 不会执行，此时按端口反查 PID，
#      但只在确认端口上服务的就是"刚跑完的这个模型"时才杀（严禁按名字 pkill——
#      本机还有别人的 vLLM，如 8001 的 teacher）
#   4. 每跑完一个就把 RESULT.txt 追加进队列汇总，全部跑完输出总表
set -uo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"

# 解释器优先级与 run.sh 一致
if [ -x "$KIT/.venv/bin/python" ]; then
  PY="$KIT/.venv/bin/python"
elif [ -x /xttest2/aiden/harness/mini-venv/bin/python ]; then
  PY=/xttest2/aiden/harness/mini-venv/bin/python
else
  echo "没有可用环境 —— 先执行：./run.sh --setup"; exit 1
fi

# ---------- 参数解析 ----------
DETACH=0; CFGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --detach) DETACH=1 ;;
    --list)
      [ -f "${2:-}" ] || { echo "--list 需要一个存在的列表文件"; exit 1; }
      while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "$line" | xargs)"
        [ -n "$line" ] && CFGS+=("$line")
      done < "$2"
      shift ;;
    *) CFGS+=("$1") ;;
  esac
  shift
done
[ "${#CFGS[@]}" -gt 0 ] || { echo "用法: run_queue.sh [--detach] CFG1.yaml CFG2.yaml ... | --list queue.txt"; exit 1; }

# 所有配置文件先验存在，避免跑了半天才发现后面的路径写错
for c in "${CFGS[@]}"; do [ -f "$c" ] || { echo "配置不存在：$c"; exit 1; }; done

# --detach：以 setsid 重新拉起自己
if [ "$DETACH" = "1" ]; then
  mkdir -p "$KIT/runs"
  LOG="$KIT/runs/queue_$(date +%Y%m%d_%H%M%S).log"
  setsid "$0" "${CFGS[@]}" >"$LOG" 2>&1 </dev/null &
  echo "队列已后台运行 pid=$! （共 ${#CFGS[@]} 个配置）—— 进度：tail -f $LOG"
  exit 0
fi

mkdir -p "$KIT/runs"
QRES="$KIT/runs/QUEUE_RESULT_$(date +%Y%m%d_%H%M%S).txt"
say() { echo "[queue $(date +%H:%M:%S)] $*"; }

# ---------- 工具函数 ----------
# 读 YAML 字段（点路径），取不到时给默认值
cfg_field() {
  "$PY" - "$1" "$2" "${3:-}" <<'PYEOF'
import sys, yaml
cur = yaml.safe_load(open(sys.argv[1])) or {}
for k in sys.argv[2].split('.'):
    cur = cur.get(k) if isinstance(cur, dict) else None
print(sys.argv[3] if cur is None else cur)
PYEOF
}

# 端口上现在服务的模型 id（无服务/超时输出空）
port_model() {
  curl -s --max-time 5 "http://localhost:$1/v1/models" 2>/dev/null \
    | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true
}

# 监听指定端口的 PID 列表（lsof 优先，ss 兜底）
port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$1" -s tcp:LISTEN 2>/dev/null || true
  else
    ss -ltnp "sport = :$1" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true
  fi
}

# 兜底清理：仅当端口上的模型 == 本轮模型时，按精确 PID 杀（TERM→等待→KILL）
cleanup_leftover() {
  local port="$1" name="$2" mid pids p
  mid="$(port_model "$port")"
  [ -z "$mid" ] && return 0                      # 端口已无服务，正常
  if [ "$mid" != "$name" ]; then
    say "警告：端口 $port 上在跑别的模型（$mid），不是本轮的 $name —— 不碰它"
    return 1
  fi
  pids="$(port_pids "$port")"
  [ -z "$pids" ] && return 0
  say "兜底清理：vLLM($name) 仍在端口 $port，按精确 PID 关闭：$pids"
  for p in $pids; do kill -15 "$p" 2>/dev/null || true; done
  for _ in $(seq 1 24); do sleep 5; [ -z "$(port_pids "$port")" ] && return 0; done
  for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
  sleep 5
}

# 等端口彻底释放（下一个 vLLM 才能绑上）
wait_port_free() {
  local port="$1" timeout="${2:-180}" waited=0
  while [ -n "$(port_pids "$port")" ]; do
    [ "$waited" -ge "$timeout" ] && { say "警告：等了 ${timeout}s 端口 $port 仍被占用"; return 1; }
    sleep 5; waited=$((waited+5))
  done
  return 0
}

# ---------- 主循环 ----------
N="${#CFGS[@]}"; i=0; FAILED=0
say "队列开始：共 $N 个配置，汇总写入 $QRES"
{ echo "swebench-kit 轮回跑汇总  $(date '+%F %T')"; echo "共 $N 个配置"; echo; } > "$QRES"

for cfg in "${CFGS[@]}"; do
  i=$((i+1))
  tag="$(cfg_field "$cfg" run.tag "")"
  name="$(cfg_field "$cfg" model.served_name "")"
  port="$(cfg_field "$cfg" serve.port 8000)"
  manage="$(cfg_field "$cfg" serve.manage True)"
  keep="$(cfg_field "$cfg" serve.keep_alive False)"

  say "── [$i/$N] $cfg  (tag=$tag  model=$name  port=$port)"
  start=$(date +%s)
  "$KIT/run.sh" "$cfg"
  rc=$?
  mins=$(( ($(date +%s) - start) / 60 ))

  # 卸载模型：正常路径 kit atexit 已关；这里只做异常兜底 + 等端口/显存释放
  if [ "$manage" = "True" ] || [ "$manage" = "true" ]; then
    if [ "$keep" = "True" ] || [ "$keep" = "true" ]; then
      say "serve.keep_alive=true，按配置保留服务，跳过卸载"
    else
      cleanup_leftover "$port" "$name" || true
      wait_port_free "$port" 180 || true
      sleep 15   # 给显存释放留余量
    fi
  fi

  # 收结果
  res="$KIT/runs/$tag/RESULT.txt"
  if [ "$rc" -eq 0 ] && [ -f "$res" ]; then
    line="$(grep -m1 RESOLVED "$res" || echo '缺 RESOLVED 行')"
    say "完成 [$i/$N] $tag：$line（${mins}min）"
    { echo "[$i/$N] OK    $tag  ${mins}min"; sed 's/^/    /' "$res"; echo; } >> "$QRES"
  else
    FAILED=$((FAILED+1))
    say "失败 [$i/$N] $tag rc=$rc —— 见 runs/$tag/kit.log，继续下一个"
    { echo "[$i/$N] FAIL  $tag  rc=$rc  ${mins}min  （runs/$tag/kit.log）"; echo; } >> "$QRES"
  fi
done

say "队列结束：$((N-FAILED))/$N 成功。汇总："
echo "────────────────────────────────────────"
cat "$QRES"
[ "$FAILED" -eq 0 ] || exit 1
