#!/usr/bin/env bash
# 评分串行队列：等各 run 的 gen 完成 -> 依次独占评分（8 并发拉满）
# 用法: setsid ./grade_queue.sh cfg1.yaml cfg2.yaml ... > queue.log 2>&1 &
set -u
export HF_ENDPOINT=https://huggingface.co
KIT=/home/ubuntu/swebench/swebench-kit
RUNS=/home/ubuntu/swebench/swebench_runs
for CFG in "$@"; do
  TAG=$(grep -m1 "tag:" "$CFG" | awk "{print \$2}")
  echo "[queue] 等待 $TAG 的 gen 完成..."
  while ! grep -q "生成完成" "$RUNS/$TAG/kit.log" 2>/dev/null; do sleep 120; done
  # 等别的评分（如 base 的自动评分）让位：sweb.eval 容器少于 3 个才开工
  while [ "$(docker ps --format "{{.Names}}" | grep -c "^sweb.eval")" -ge 3 ]; do sleep 120; done
  echo "[queue] $TAG 开始评分 $(date)"
  cd "$KIT" && ./run.sh "$CFG" --stages grade
  echo "[queue] $TAG 评分结束 $(date): $(grep RESOLVED $RUNS/$TAG/RESULT.txt 2>/dev/null)"
done
echo "[queue] 全部完成 $(date)"
