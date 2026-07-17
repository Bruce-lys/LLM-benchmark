#!/usr/bin/env bash
# ============================================================
# 打包 SWE-bench 评测环境，用于迁移到 CPU-only 服务器
#
# 用法（在源机器上、kit 目录内执行）:
#   ./migrate/pack_eval_env.sh                     # 只打包代码+数据+清单（小，~100M）
#   ./migrate/pack_eval_env.sh --with-images lite75  # 连 75 个评测镜像一起打包（~80-150G，慢）
#   ./migrate/pack_eval_env.sh --out /xttest2/hera   # 指定输出目录
#
# 产物: <out>/swebench-eval-pack.tar.gz
#       （含 setup_eval_env.sh，到目标机解包后执行它即可）
# 镜像默认不打包：目标机能连 registry 时用清单现拉更省事（setup 脚本支持）。
# ============================================================
set -euo pipefail

KIT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$KIT/.."
WITH_IMAGES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --with-images) WITH_IMAGES="${2:-lite75}"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

STAGE="$OUT/swebench-eval-pack"
rm -rf "$STAGE"; mkdir -p "$STAGE"

echo "[1/5] 复制 kit 代码与数据（不含 runs/.venv）..."
rsync -a --exclude=runs --exclude=.venv --exclude=__pycache__ "$KIT/" "$STAGE/kit/"

echo "[2/5] 生成镜像清单..."
munge() { sed 's/__/_1776_/g'; }
# lite75 清单
munge < "$KIT/data/lite75_ids.txt" \
  | sed 's#^#swebench/sweb.eval.x86_64.#; s#$#:latest#' \
  > "$STAGE/images_lite75.txt"
# verified 全量清单（从数据集 jsonl 提 instance_id）
python3 - "$KIT/data/verified_test.jsonl" > "$STAGE/images_verified.txt" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    iid = json.loads(line)["instance_id"].replace("__", "_1776_")
    print(f"swebench/sweb.eval.x86_64.{iid}:latest")
PY
echo "  lite75: $(wc -l < "$STAGE/images_lite75.txt") 个; verified: $(wc -l < "$STAGE/images_verified.txt") 个"

echo "[3/5] 放入安装脚本..."
cp "$KIT/migrate/setup_eval_env.sh" "$STAGE/"
chmod +x "$STAGE/setup_eval_env.sh"

if [ -n "$WITH_IMAGES" ]; then
  echo "[4/5] 导出 docker 镜像（$WITH_IMAGES，很慢很大，可随时 Ctrl-C 改用清单现拉）..."
  mkdir -p "$STAGE/images"
  M="$STAGE/images_${WITH_IMAGES}.txt"
  [ -f "$M" ] || { echo "无清单 $M"; exit 1; }
  n=0; total=$(wc -l < "$M")
  while read -r img; do
    n=$((n+1))
    f="$STAGE/images/$(echo "$img" | tr '/:' '__').tar.gz"
    [ -s "$f" ] && { echo "  ($n/$total) 已存在，跳过 $img"; continue; }
    echo "  ($n/$total) docker save $img"
    docker save "$img" | gzip > "$f.tmp" && mv "$f.tmp" "$f"
  done < "$M"
else
  echo "[4/5] 跳过镜像导出（目标机将按清单在线拉取）"
fi

echo "[5/5] 打 tar 包..."
TARBALL="$OUT/swebench-eval-pack.tar.gz"
tar -C "$OUT" -czf "$TARBALL" swebench-eval-pack
du -sh "$TARBALL"
echo "完成。传到目标机后："
echo "  tar xzf swebench-eval-pack.tar.gz && cd swebench-eval-pack && ./setup_eval_env.sh --workroot /data/swebench"
