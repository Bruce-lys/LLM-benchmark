# SWE-bench 评测步骤（B300 / 普通账号）

流水线：vLLM serve → gen（agent 在 docker 容器里跑题）→ grade（docker 评分）→ RESULT.txt。
耗时参考（lite75，64 并发，2 卡）：加载 8-20 分钟，生成 1-2 小时，评分 20-40 分钟。

## 1. 前置检查

```bash
docker ps                     # 报 permission denied -> 让管理员: sudo usermod -aG docker <账号>，重新登录
df -h /                       # 根盘可用 <20G 时清理: docker builder prune -af && docker image prune -f
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 确认有 >=2 张空闲卡
ss -ltn | grep 8302           # 无输出 = 端口可用；被占则换一个
```

## 2. 环境变量（每个 shell 都要 source）

存成 `~/swebench_env.sh`：

```bash
export WORKROOT=/xttest2/<你的名字>       # 你的工作目录
mkdir -p $WORKROOT/tmp $WORKROOT/.config $WORKROOT/.cache
export HOME=$WORKROOT
export TMPDIR=$WORKROOT/tmp
export XDG_CONFIG_HOME=$WORKROOT/.config
export XDG_CACHE_HOME=$WORKROOT/.cache
export PYTHONPATH=$WORKROOT/pyshim        # 步骤 4 的 shim
```

验证：`python3 -c "import tempfile; print(tempfile.gettempdir())"` 输出 `$WORKROOT/tmp`。

## 3. 复制 kit + 建 venv（一次性）

```bash
source ~/swebench_env.sh
rsync -a --exclude=runs --exclude=.venv --exclude=__pycache__ \
      /xttest2/swebench-kit/ $WORKROOT/swebench-kit/
cd $WORKROOT/swebench-kit

python3 -m venv .venv
M=https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/pip install --no-cache-dir -i $M --upgrade pip
.venv/bin/pip install --no-cache-dir -i $M "pyyaml>=6"
.venv/bin/pip install --no-cache-dir -i $M "swebench==4.0.4"
.venv/bin/pip install --no-cache-dir -i $M "mini-swe-agent==2.4.2"
```

版本必须锁死（换版本 = 换口径）。验证：

```bash
.venv/bin/python -c "import minisweagent, swebench, yaml; print('imports OK')"
ls .venv/bin/mini-extra
```

## 4. flashinfer shim（一次性，vLLM serve 必须）

```bash
mkdir -p $WORKROOT/flashinfer_local
cp -a /xttest/grpo/envs/vllm20/lib/python3.11/site-packages/flashinfer_cubin/cubins \
      $WORKROOT/flashinfer_local/cubins
chmod -R u+w $WORKROOT/flashinfer_local/cubins

V=$(/xttest/grpo/envs/vllm20/bin/python -c "import flashinfer_cubin as f; print(f.__version__)")
mkdir -p $WORKROOT/pyshim/flashinfer_cubin
cat > $WORKROOT/pyshim/flashinfer_cubin/__init__.py <<EOF
__version__ = "$V"
def get_cubin_dir():
    return "$WORKROOT/flashinfer_local/cubins"
EOF
```

验证（最后一行必须是 LOCK-WRITE-OK）：

```bash
PYTHONPATH=$WORKROOT/pyshim /xttest/grpo/envs/vllm20/bin/python -c "
from flashinfer.jit.env import FLASHINFER_CUBIN_DIR; print(FLASHINFER_CUBIN_DIR)
import os; p=str(FLASHINFER_CUBIN_DIR)+'/flashinfer/write_test.lock'
open(p,'w').close(); os.remove(p); print('LOCK-WRITE-OK')"
```

## 5. 写配置

`$WORKROOT/configs/my_lite75.yaml`：

```yaml
run:
  tag: <用户名>_<模型>_<数据集>_<口径>   # 用户名开头 + 全局唯一，绝不复用
  output_root: /xttest2/<你>/swebench_runs
  dataset: lite75            # verified | swelite | lite75
  workers: 64
  max_passes: 3
  retry_empty: true          # 对比实验两侧必须一致
model:
  path: /xttest2/models/Qwen3.6-35B-A3B
  served_name: Qwen3.6-35B-A3B
sampling:
  temperature: 0.85          # 团队 lite75 口径 0.85；历史 full-500 锚点 1.0
  top_p: 0.95
agent:
  step_limit: 250
serve:
  manage: true
  endpoint: http://localhost:8302/v1
  port: 8302
  gpus: 2                    # 自动挑空闲卡，不会碰别人占的卡
  tensor_parallel: auto
  extra_args: ["--limit-mm-per-prompt", '{"image":0,"video":0}']   # 必带
grade:
  enabled: true
  rolling: false
  max_workers: 24
  namespace: swebench
```

试跑 5 题：run 段加 `slice: "0:5"`。

## 6. 启动

```bash
source ~/swebench_env.sh
cd $WORKROOT/swebench-kit
./run.sh $WORKROOT/configs/my_lite75.yaml --dry-run    # 检查渲染出的命令
./run.sh $WORKROOT/configs/my_lite75.yaml --detach     # 正式启动，断 SSH 不影响
```

## 7. 监控（run_dir = output_root/tag）

```bash
tail -f <run_dir>/kit.log                              # 总控：serve 就绪 / 第 N 轮 / 评分
tail -f <run_dir>/vllm.log                             # serve 失败原因搜 ERROR / PermissionError
find <run_dir>/gen -name "*.traj.json" | wc -l         # 完成轨迹数
docker ps | wc -l                                      # 生成期 ≈ workers 个容器
```

## 8. 出分

- `<run_dir>/RESULT.txt` —— 总分 + 口径戳（报分时口径一起报）
- `<run_dir>/grade/` —— 逐题报告
- `<run_dir>/grade/touches_tests.txt` —— patch 改了测试文件的题，报分前过一眼

## 9. 断点补救

```bash
./run.sh 配置 --detach            # 中断后重跑：已完成轨迹自动跳过
./run.sh 配置 --stages grade      # 生成完但评分失败：补评
```

## 10. 故障速查

| 症状 | 修法 |
|---|---|
| `No usable temporary directory` / 写 `~` 失败 | 没 source 步骤 2 的环境 |
| pip 极慢 | 加 `-i` 清华源（步骤 3） |
| vLLM 加载崩，vllm.log 有 `PermissionError ... .lock` | 步骤 4 shim 没生效；查 vLLM 进程环境里有无 PYTHONPATH |
| `permission denied ... docker.sock` | 管理员加 docker 组，重新登录 |
| 容器 ENOSPC | 清根盘（步骤 1） |
| 评分 docker 409 conflict | `docker ps -aq --filter "name=sweb.eval.*.<你的tag>" \| xargs -r docker rm -f` |
| RESULT 分母不对 | tag 混了不同实验的轨迹，换新 tag 重跑 |
| JMS 报 `match asset failed` | JumpServer 抽风，重试 |

## 11. 共享机器规矩

- 不碰不属于自己的 GPU；杀进程只按精确 PID，禁止 pkill 按名字杀。
- tag 不复用；产物只写自己的 output_root；共享 kit 目录只读。
