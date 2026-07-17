#!/usr/bin/env bash
# ============================================================
# 在 CPU-only 服务器上恢复 SWE-bench 评测环境
# （由 pack_eval_env.sh 打出的包内自带；在解包后的目录里执行）
#
# 用法:
#   ./setup_eval_env.sh --workroot /data/swebench            # 基本安装
#   ./setup_eval_env.sh --workroot /data/swebench --pull-images lite75   # 顺便在线拉镜像
#   ./setup_eval_env.sh --workroot /data/swebench --load-images          # 从包内 images/ 离线导入
#
# 完成后目录结构:
#   $WORKROOT/swebench-kit/     kit + 自建 venv（已打好网络补丁）
#   $WORKROOT/env.sh            每个 shell 先 source 它
#   $WORKROOT/configs/          示例配置（连远端 GPU 服务的 manage:false 模式）
# ============================================================
set -euo pipefail

PACK="$(cd "$(dirname "$0")" && pwd)"
WORKROOT=""
PULL_IMAGES=""
LOAD_IMAGES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --workroot) WORKROOT="$2"; shift 2;;
    --pull-images) PULL_IMAGES="${2:-lite75}"; shift 2;;
    --load-images) LOAD_IMAGES=1; shift;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done
[ -n "$WORKROOT" ] || { echo "必须指定 --workroot <目录>"; exit 1; }

echo "===== [0] 前置检查 ====="
[ "$(uname -m)" = "x86_64" ] || { echo "FATAL: 本机是 $(uname -m)，官方 swebench 镜像只有 amd64，换 x86_64 机器"; exit 1; }
docker ps >/dev/null 2>&1 || { echo "FATAL: docker 不可用（没装或没权限）。让管理员: usermod -aG docker $(whoami)"; exit 1; }
command -v python3 >/dev/null || { echo "FATAL: 没有 python3"; exit 1; }
PYV=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  arch=x86_64 OK; docker OK; python3=$PYV"
AVAIL=$(df -Pk "$(dirname "$WORKROOT")" 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || echo "?")
echo "  磁盘可用 ~${AVAIL}G（lite75 镜像约需 100G，全量 verified 约 400G，注意 docker data-root 所在盘）"

echo "===== [1] 安装 kit ====="
mkdir -p "$WORKROOT"
rsync -a "$PACK/kit/" "$WORKROOT/swebench-kit/"
cd "$WORKROOT/swebench-kit"

echo "===== [2] 建 venv（多镜像源自动回退）====="
python3 -m venv .venv
try_pip() {  # try_pip <包...>
  for M in "https://pypi.tuna.tsinghua.edu.cn/simple" "https://mirrors.aliyun.com/pypi/simple" ""; do
    idx=(); [ -n "$M" ] && idx=(-i "$M")
    echo "  pip install $* ${M:+via $M}"
    if .venv/bin/pip install --no-cache-dir "${idx[@]}" "$@"; then return 0; fi
  done
  return 1
}
.venv/bin/pip -q install --upgrade pip 2>/dev/null || true
try_pip "pyyaml>=6"
try_pip "swebench==4.0.4"          # 版本锁死 = 口径锁死，勿改
try_pip "mini-swe-agent==2.4.2"

echo "===== [3] 打 swebench 网络补丁（GitHub raw 缓存+重试）====="
SPFILE=$(.venv/bin/python -c "import swebench.harness.test_spec.python as m; print(m.__file__)")
if grep -q "swebench-kit patch" "$SPFILE"; then
  echo "  已打过，跳过"
else
  cat >> "$SPFILE" <<EOF


# === swebench-kit patch: GitHub raw 缓存 + 重试（弱网环境下评分零产出的救命补丁）===
import os as _os, time as _time
_REQS_CACHE = "$WORKROOT/swebench-kit/data/reqs_cache"
def _cached_retry(fn):
    def wrap(*args, **kw):
        key = fn.__name__ + "__" + "__".join(str(a).replace("/", "_") for a in args)
        path = _os.path.join(_REQS_CACHE, key)
        if _os.path.exists(path):
            with open(path) as f:
                return f.read()
        last = None
        for i in range(8):
            try:
                out = fn(*args, **kw)
                _os.makedirs(_REQS_CACHE, exist_ok=True)
                tmp = path + ".tmp%d" % _os.getpid()
                with open(tmp, "w") as f:
                    f.write(out)
                _os.replace(tmp, path)
                return out
            except Exception as e:
                last = e
                _time.sleep(5 * (i + 1))
        raise last
    return wrap
get_requirements_by_commit = _cached_retry(get_requirements_by_commit)
get_environment_yml_by_commit = _cached_retry(get_environment_yml_by_commit)
EOF
  echo "  补丁已写入 $SPFILE（缓存目录随包带来，常用条目已预热）"
fi

echo "===== [4] 验证 ====="
.venv/bin/python -c "import minisweagent, swebench, yaml; print('  imports OK')"
[ -x .venv/bin/mini-extra ] && echo "  mini-extra OK"
.venv/bin/python -c "import swebench.harness.test_spec.python as m; print('  patch OK:', m._REQS_CACHE)"

echo "===== [5] 环境脚本与示例配置 ====="
cat > "$WORKROOT/env.sh" <<EOF
# source 我，再跑 kit
export WORKROOT=$WORKROOT
# 健康机器无需重定向 HOME；若本机根盘紧张/家目录受限，取消下面注释:
# mkdir -p \$WORKROOT/tmp \$WORKROOT/.config \$WORKROOT/.cache
# export HOME=\$WORKROOT TMPDIR=\$WORKROOT/tmp XDG_CONFIG_HOME=\$WORKROOT/.config XDG_CACHE_HOME=\$WORKROOT/.cache
EOF
mkdir -p "$WORKROOT/configs" "$WORKROOT/swebench_runs"
cat > "$WORKROOT/configs/example_remote_gpu.yaml" <<EOF
# CPU 评测机模式：模型服务在别的 GPU 机器上，本机只跑 gen+grade
run:
  tag: <用户名>_<模型>_lite75_t085        # 用户名开头、全局唯一、绝不复用
  output_root: $WORKROOT/swebench_runs
  dataset: lite75
  workers: 32                              # 按本机核数调；只影响速度不影响分数
  max_passes: 3
  retry_empty: true
model:
  path: /dev/null                          # manage:false 时不加载权重，占位即可
  served_name: <GPU机器上served的模型名>    # 必须与远端一致，kit 会校验
sampling:
  temperature: 0.85
  top_p: 0.95
agent:
  step_limit: 250
serve:
  manage: false                            # 不起本地服务
  endpoint: http://<GPU机器IP>:8302/v1     # 远端 vLLM；跨网段可用 ssh -L 隧道
grade:
  enabled: true
  rolling: false
  max_workers: 16                          # 按本机核数调
  namespace: swebench
EOF

if [ "$LOAD_IMAGES" = 1 ]; then
  echo "===== [6] 离线导入镜像 ====="
  for f in "$PACK"/images/*.tar.gz; do
    [ -e "$f" ] || { echo "  包内无 images/，跳过"; break; }
    echo "  docker load < $(basename "$f")"
    gunzip -c "$f" | docker load
  done
elif [ -n "$PULL_IMAGES" ]; then
  echo "===== [6] 在线拉取镜像（$PULL_IMAGES，失败自动重试 3 次）====="
  M="$PACK/images_${PULL_IMAGES}.txt"
  [ -f "$M" ] || { echo "无清单 $M"; exit 1; }
  fail=0
  while read -r img; do
    ok=0
    for i in 1 2 3; do docker pull -q "$img" && { ok=1; break; } || sleep 10; done
    [ "$ok" = 1 ] || { echo "  PULL-FAIL: $img"; fail=$((fail+1)); }
  done < "$M"
  echo "  拉取完成，失败 $fail 个（失败的评分时还会自动重试一次）"
else
  echo "===== [6] 未处理镜像。之后可用: xargs -a $PACK/images_lite75.txt -n1 docker pull ====="
fi

echo ""
echo "===== 完成。下一步 ====="
echo "  1) source $WORKROOT/env.sh"
echo "  2) 编辑 $WORKROOT/configs/example_remote_gpu.yaml（endpoint/served_name/tag）"
echo "  3) 在 GPU 机器上把 vLLM 起好（或让 kit 在那边 --stages serve + keep_alive:true）"
echo "  4) cd $WORKROOT/swebench-kit && ./run.sh $WORKROOT/configs/<你的>.yaml --dry-run 检查后 --detach"
echo "详细步骤见 $WORKROOT/swebench-kit/README_hera.md"
