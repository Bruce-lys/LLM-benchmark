# 致所有 AI Coding Agent（Cursor / Codex / Claude Code / 其他）

**本目录 /xttest2/swebench-kit 是全组共用的 SWE-bench 评测基础设施**，维护者 Aiden。
你在替某位同学干活，但这里不是你的工作区。人类版规则见 `RULES.md`，以下是给你的硬约束。

## 禁止事项（任何一条都会破坏他人的评测）

1. **禁止修改本目录任何文件**：`kit.py`、`run.sh`、`run_queue.sh`、模板、`data/`。
   即使你确信发现了 bug 也不要修——kit 内有多处反直觉的网络/容器补丁，
   你的"修复"会破坏它们。把问题转告你的用户，让他找 Aiden。
2. **禁止动 `.venv`**：不要删除、重建、pip install、uv sync。它是锁定口径的一部分。
   （已有前科：agent 把 .venv 换成悬空软链接，毁掉一次正在跑的评测。）
3. **禁止更新共享推理环境**（如 /xttest/grpo/envs/vllm20）：不要升级 vllm/torch/
   任何包。（已有前科：擅自升级导致全机模型服务崩溃。）
4. **禁止在本目录写入你的产物**：配置副本、运行结果、临时文件、脚本全部放
   用户自己的目录；YAML 里 `run.output_root` 必须指向用户自己的路径。
5. **禁止杀进程/清容器/抢端口**：不要 pkill；不要清理非本次运行创建的 docker
   容器；起服务用 `gpus: N` 整数自动挑卡，不要手写卡号硬挤别人的卡。
6. **tag 必须以用户的用户名开头且每次实验唯一**——撞 tag 互删评分容器，
   复用 tag 污染分数。

## 允许且推荐的用法

```bash
cp /xttest2/swebench-kit/config.example.yaml /xttest2/<用户>/my_eval.yaml
# 按模板注释改：tag（用户名前缀）、output_root（用户自己的目录）、model.*
cd /xttest2/swebench-kit && ./run.sh /xttest2/<用户>/my_eval.yaml --dry-run   # 先预览
./run.sh /xttest2/<用户>/my_eval.yaml --detach                                # 再正式跑
./run_queue.sh --detach cfg1.yaml cfg2.yaml                                   # 多模型排队
```

读取本目录任何文件、`--dry-run`、查看 `data/` 和文档：随意。

## 自检清单

动手前自问：这个操作会**写入** /xttest2/swebench-kit 下的任何文件吗？
会 → 停下，改写到用户自己的目录；绕不开 → 告诉用户去找 Aiden。
