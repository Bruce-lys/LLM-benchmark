# swebench-kit（共享评测台）

一个 YAML 配置跑通 SWE-bench 完整管线：**vLLM 起服务 → mini-swe-agent 生成轨迹 →
官方 swebench 评分 → RESULT.txt**。由 Aiden 维护；使用规则见 `RULES.md`（必读）。

## 快速开始

```bash
# 首次（或 .venv 损坏时，仅维护者执行）：构建锁定版本的评测环境
./run.sh --setup

# 1. 把模板复制到你自己的目录并按注释修改（tag / output_root / model 必改）
cp /xttest2/swebench-kit/config.example.yaml /xttest2/<你>/my_eval.yaml

# 2. 预览将执行的命令（不实际运行）
./run.sh /xttest2/<你>/my_eval.yaml --dry-run

# 3. 正式跑（--detach 后台运行，ssh 断开不影响）
./run.sh /xttest2/<你>/my_eval.yaml --detach

# 4. 多模型排队：跑完一个自动卸载、加载下一个，逐个出分并汇总
./run_queue.sh --detach /xttest2/<你>/m1.yaml /xttest2/<你>/m2.yaml
```

结果在 `<output_root>/<tag>/RESULT.txt`，戳有完整口径（dataset/temp/steps/retry_empty）。

## 数据集

| 名字 | 题数 | 定位 |
|---|---|---|
| `verified` | 500 | SWE-bench Verified 官方全量。**唯一可对外报告、可对标官方 leaderboard 的数据集** |
| `swelite` | 100 | 基于 27B/35B 和 Ornith 的弱点合成的数据集（65 题模型稳定失败 + 35 题通过），**难度显著偏高，分数不可作为模型能力参考**，仅用于弱点针对性分析 |
| `lite75` | 75 | 按 15 简单 + 45 中等 + 15 难配比，难度标注分布与全量 500 接近（略偏易）。**用途：快速验证模型能力、预测 full-500 Verified 分数区间，作为开发期参考**；正式结论仍以 full-500 为准。选题口径逐题可溯源：`data/lite75_manifest.tsv`，生成脚本 `data/make_lite75.py` |

`data/verified_test.jsonl` 含 OpenAI 人工难度标注（`difficulty` 字段），
全量分布：<15min 38.8%，15min-1h 52.2%，1-4h 8.4%，>4h 0.6%。
参考锚点（lite75，temp1.0/step250/单遍）：Qwen3.6-35B-A3B base = 55/75。

## 官方一致性声明（2026-07-03 审计）

以下三层已逐项与官方实测比对，评测判定与官方完全一致：

1. **数据**：本地 500 题与 HF `princeton-nlp/SWE-Bench_Verified` 逐字段一致
   （金标 patch / test_patch / FAIL_TO_PASS / PASS_TO_PASS / base_commit /
   难度标注，500/500 全对上）。
2. **评分器**：官方 swebench 4.0.4 + 官方预构建镜像（namespace `swebench`）+
   官方判定标准（F2P 全过且 P2P 全过），判分逻辑零改动。kit 仅有两个非语义
   补丁：报告阶段网络挂起修复、评分前残留容器清理。
3. **脚手架提示词**：与 mini-swe-agent（SWE-bench 官方团队出品）内置 swebench
   模板逐字节一致；改动仅限运维参数（命令超时、输出截断、断流重试提示）。

**对外报分的披露模板**（脚手架/预算/采样是 leaderboard 各家自选项，必须写明）：

> XX%（SWE-bench Verified full-500，mini-swe-agent 脚手架，step_limit 250，
> temp 1.0 / top_p 0.95，pass@1，retry_empty=off，engine=vX.Y.Z）

注意：对外报"严格 pass@1"时把 `retry_empty` 显式设为 false（默认 on 是内部
开发口径，空题复测≈+3pp）；引擎版本以 RESULT.txt 的 engine 戳为准。

## 口径要点（对比实验必读）

1. **RESULT.txt 的戳就是口径**：temp / steps / retry_empty / dataset 任何一项不同的
   两个分数不可直接对比。官方口径：temp 1.0 / top_p 0.95 / step250 / retry_empty on。
2. **run variance ±5**：单次运行的分差 <5 题不构成结论，关键对比要多跑取均值。
3. **引擎版本**：vllm20 环境 2026-07-02 从 v0.20.0 被升到 v0.22.1，跨引擎的分数
   可能有 1-2 个点漂移，严谨对比需同引擎重跑基线。
4. **空 patch 率**：分数异常低时先查 `gen/exit_statuses_*.yaml`——InternalServerError
   是服务事故（换卡重跑），TimeoutExpired/LimitsExceeded 是预算耗尽，都不是能力分。

## 内置的坑修复（改代码前先读这里）

- swebench 4.0.4 评分报告阶段对 GitHub 的无超时请求在国内网络会永久挂起——kit
  已打补丁（client=None + 60s socket 超时）。
- 中断的评分会留下同名容器导致重评 409 报错——kit 评分前自动清理同 tag 容器
  （这也是 tag 必须唯一的原因）。
- 触碰测试文件的 patch 自动记录在 `grade/touches_tests.txt`（test-gaming 审计）。

## 目录结构

```
swebench-kit/
├── kit.py              # 主程序（DEFAULTS 里有全部参数的默认值和注释）
├── run.sh              # 入口：--setup / --dry-run / --detach / --stages
├── run_queue.sh        # 多模型轮回跑
├── config.example.yaml # 配置模板（每个参数都有注释）
├── data/               # 数据集与子集定义（只读）
├── runs/               # 仅维护者冒烟测试用；你的产物写自己的 output_root
├── RULES.md            # 使用规则（人读）
└── AGENTS.md           # AI coding agent 行为规则（CLAUDE.md/.cursorrules 同文件）
```
