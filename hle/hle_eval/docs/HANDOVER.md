# HLE 评测交接说明

本文面向接手 HLE（Humanity's Last Exam）模型评测的同学，说明本仓库的评测流程、用到的脚本，以及如何换模型跑通全流程。

官方数据集与说明：[cais/hle](https://huggingface.co/datasets/cais/hle) · [lastexam.ai](https://lastexam.ai/)

---

## 1. HLE 是什么、我们怎么评

### 1.1 背景

HLE 是面向前沿大模型的学术闭卷基准，约 2500 题，覆盖数学、物理、生物、人文等学科，题型含**简答（exactMatch）**与**多选（multipleChoice）**。

官方推荐两阶段：

1. **模型预测**：按固定格式输出 `Explanation / Answer / Confidence`
2. **LLM Judge**：用 Judge 模型对比标准答案，输出 `correct: yes/no`，并计算 Accuracy 与 Calibration Error

本仓库在此基础上做了工程化改造，便于在 **SGLang OpenAI 兼容接口**上批量测多个本地/远端模型。

### 1.2 本仓库的实际做法（当前默认）

| 项目 | 本仓库默认 |
|------|------------|
| 数据集 | HuggingFace `cais/hle`（test split） |
| 评测子集 | **10% 分层抽样**，按 `category × difficulty(answer_type)`，250 题 |
| 推理入口 | SGLang `http://IP:PORT/v1` |
| 输出格式 | 强制 `Explanation / Answer / Confidence` |
| 采样参数 | `temperature=0.85, top_p=0.95, top_k=20, ...`（见 YAML） |
| 打分 | **Realtime Judge**：每题推理完立即用 DeepSeek 按官方 prompt 判分 |
| 并发 | 多模型并行（`parallel`）；单模型内默认 `num_workers=1` |
| 断点续跑 | 已写入 `outputs/predictions/*.json` 的题会自动跳过 |

> **与官方 leaderboard 的差异**：子集（非全量）、Judge 模型（DeepSeek，非 o3-mini）、temperature（0.85，非 0）。  
> 本流程适合**同条件横向对比多个融合模型**；若要对齐官方分数，需另用全量 + o3-mini（见文末）。

### 1.3 评分口径（汇报时请统一）

- **分子**：`judge_response.correct == "yes"` 的题数  
- **分母（推荐）**：有落盘 `response` 的题数（**超时未写入 JSON，自然排除**）  
- 准确率保留 1 位小数用于榜单

示例：`准确率 = 正确数 / 已做题数`

---

## 2. 目录结构

```text
hle/                              # 仓库根
├── requirements.txt
├── README.md                     # 上游 HLE 简介
└── hle_eval/                     # ★ 评测工程（日常只用这里）
    ├── run_hle.sh                # 一键入口（给使用者）
    ├── README.md                 # 快速上手
    ├── configs/
    │   ├── default.yaml          # 默认参数说明
    │   ├── models/
    │   │   ├── example.yaml      # 换模型模板（复制它）
    │   │   └── current.yaml      # 当前评测目标
    │   └── legacy/               # 旧 JSON 配置（兼容）
    ├── src/                      # Python 脚本
    ├── data/subsets/             # 固定子集 ID
    ├── outputs/predictions/      # 模型输出 JSON（含 reasoning + judge）
    └── docs/
        ├── HANDOVER.md           # 本文档
        └── sglang_hle_commands.md
```

---

## 3. 脚本一览（谁干什么）

### 3.1 平时必用

| 脚本 | 作用 |
|------|------|
| `run_hle.sh` | **对外入口**。读 YAML/JSON，调用批量评测 |
| `src/run_sglang_batch.py` | 按配置并发启动多个模型评测进程 |
| `src/run_model_predictions.py` | **单模型核心**：加载子集 → 调模型推理 → realtime Judge → 落盘 |
| `src/config_loader.py` | 加载/合并 YAML·JSON，解析子集与输出路径 |
| `src/judge_prompts.py` | **官方风格 Judge prompt**（与 `run_judge_results` 共用） |
| `src/subset_utils.py` | 子集读写、分层抽样工具 |

### 3.2 按需使用

| 脚本 | 何时用 |
|------|--------|
| `src/check_sglang_thinking.py` | 新 endpoint 上线时，确认返回了 `reasoning_content` |
| `src/create_hle_subset.py` | 重新生成分层子集（一般不用，已有固定 250 题文件） |
| `src/run_judge_results.py` | **事后两阶段 Judge**（对已有预测重打官方指标 / Calibration Error） |

### 3.3 数据文件

| 文件 | 作用 |
|------|------|
| `data/subsets/hle_subset_10pct_category_difficulty_all.json` | 当前固定 250 题子集（category + difficulty 分层） |
| `outputs/predictions/hle_<model>.json` | 每个模型的预测 + Judge 结果 |
| `configs/models/*.yaml` | 模型列表与全局评测参数 |

---

## 4. 端到端评测流程

```text
① 模型服务已启动（SGLang /v1）
        ↓
② 确认 /v1/models 中的 model id
        ↓
③ 写 YAML（name + base_url）
        ↓
④ （可选）check_sglang_thinking.py
        ↓
⑤ ./run_hle.sh configs/models/xxx.yaml
        ↓
⑥ 查看 outputs/predictions/hle_xxx.json
        ↓
⑦ 按「正确/已做」统计准确率
```

### Step A. 环境准备（每人一次）

```bash
cd /path/to/hle
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=EMPTY              # SGLang 一般不校验，但 SDK 需要非空
export DEEPSEEK_API_KEY=sk-你的密钥      # realtime Judge 必需
```

### Step B. 确认模型服务

```bash
curl -s http://<IP>:<PORT>/v1/models | python3 -m json.tool
```

记下返回的 `"id"`，YAML 里的 `name` **必须与此一致**。

（可选）验证 thinking：

```bash
cd hle_eval
PYTHONPATH=src python src/check_sglang_thinking.py \
  --base_url http://<IP>:<PORT>/v1 \
  --model <MODEL_ID>
```

期望看到 `reasoning_content_present: True`。

### Step C. 配置模型

```bash
cd hle_eval
cp configs/models/example.yaml configs/models/my_model.yaml
```

编辑 `my_model.yaml`，最少改：

```yaml
parallel: 1
models:
  - name: <MODEL_ID>                 # 与 /v1/models 的 id 一致
    base_url: http://<IP>:<PORT>/v1
```

其余默认已齐（子集、采样、Judge）。常见可选项：

```yaml
defaults:
  timeout: 1800        # 长链推理容易超时可调大（秒）
  num_workers: 1       # 单模型并发请求数；显存紧张保持 1

parallel: 2            # 同时测几个模型就写几
```

多模型示例：

```yaml
parallel: 2
models:
  - name: model_a
    base_url: http://10.0.3.50:8000/v1
  - name: model_b
    base_url: http://10.0.3.51:8000/v1
```

### Step D. 运行评测

```bash
cd hle_eval

# 1）先 dry-run，确认命令与路径正确
./run_hle.sh configs/models/my_model.yaml --dry-run

# 2）smoke：只跑前 3 题
./run_hle.sh configs/models/my_model.yaml --max_samples 3

# 3）正式：250 题子集（断点续跑）
./run_hle.sh configs/models/my_model.yaml
```

推荐用 tmux，避免 SSH 断开：

```bash
tmux new -s hle_eval
./run_hle.sh configs/models/my_model.yaml 2>&1 | tee run_my_model.log
# Ctrl-b d 分离
```

### Step E. 看结果

输出文件：`outputs/predictions/hle_<MODEL_ID>.json`

每条大致结构：

```json
{
  "<question_id>": {
    "model": "...",
    "response": "Explanation: ...\nAnswer: ...\nConfidence: ...",
    "reasoning_content": "...",
    "usage": {"prompt_tokens": ..., "completion_tokens": ...},
    "judge_response": {
      "correct_answer": "...",
      "model_answer": "...",
      "reasoning": "...",
      "correct": "yes|no",
      "confidence": 90,
      "judge_model": "deepseek-v4-flash"
    }
  }
}
```

临时统计准确率（分母=有 response 的题）：

```bash
cd hle_eval
python3 - <<'PY'
import json
from pathlib import Path

subset = set(json.load(open("data/subsets/hle_subset_10pct_category_difficulty_all.json"))["ids"])
path = Path("outputs/predictions/hle_YOUR_MODEL.json")  # 改文件名
data = json.load(open(path))
items = [v for k, v in data.items() if k in subset and v.get("response")]
yes = sum(1 for v in items if (v.get("judge_response") or {}).get("correct") == "yes")
print(f"{path.name}: {yes}/{len(items)} = {100*yes/len(items):.1f}%")
PY
```

---

## 5. 配置项说明（YAML）

| 字段 | 含义 |
|------|------|
| `dataset` | HF 数据集，默认 `cais/hle` |
| `subset_file` | 子集文件路径 |
| `parallel` | 同时跑几个模型进程 |
| `realtime_judge` | 是否每题立即 Judge |
| `defaults.*` | 采样与超时等，单模型可覆盖 |
| `judge.*` | Judge 模型 / URL / env / timeout |
| `models[].name` | 推理服务上的模型名 |
| `models[].base_url` | OpenAI 兼容 API 根路径（含 `/v1`） |
| `models[].output` | 可选；默认 `outputs/predictions/hle_<name>.json` |

CLI 可覆盖部分项，例如：

```bash
./run_hle.sh configs/models/current.yaml --parallel 1
./run_hle.sh configs/models/current.yaml --only Ornith-0.9_Qwen3.6-0.1
./run_hle.sh configs/models/current.yaml --max_samples 3
./run_hle.sh configs/models/current.yaml --timeout 1800
```

---

## 6. 固定子集说明

当前使用：

`data/subsets/hle_subset_10pct_category_difficulty_all.json`

- 全量中抽 **约 10%**
- 分层字段：`category` + `difficulty`（映射自数据集 `answer_type`：`exactMatch` / `multipleChoice`）
- `seed=42`，可复现
- `total_ids ≈ 250`

**换模型时务必使用同一子集文件**，否则分数不可比。

若需重建（谨慎）：

```bash
cd hle_eval
PYTHONPATH=src python src/create_hle_subset.py \
  --ratio 0.1 \
  --seed 42 \
  --stratify_by category,difficulty \
  --output data/subsets/hle_subset_10pct_category_difficulty_all.json
```

---

## 7. 与官方完整评测的关系

| | 本仓库默认（横向对比） | 更接近官方 |
|--|------------------------|------------|
| 题量 | 250 题子集 | 全量 ~2500 |
| temperature | 0.85 | 0 |
| Judge | DeepSeek realtime | `o3-mini-2025-01-31` |
| Calibration Error | realtime 结果含 confidence，可后算 | `run_judge_results.py` 正式输出 |

事后官方风格 Judge（需自行改 judge 模型密钥与参数）：

```bash
cd hle_eval
PYTHONPATH=src python src/run_judge_results.py \
  --dataset cais/hle \
  --predictions outputs/predictions/hle_MODEL.json \
  --subset_file data/subsets/hle_subset_10pct_category_difficulty_all.json \
  --judge o3-mini-2025-01-31 \
  --judge_provider openai \
  --num_workers 20 \
  --output outputs/predictions/judged_hle_MODEL.json
```

注意：若预测 JSON 里已有 `judge_response`，该脚本会**跳过**已判题。要对官方 Judge 重判，需先去掉 `judge_response` 字段或输出到新文件流程。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|------|-------------|
| `Model not found` | YAML `name` 与 `/v1/models` 的 id 不一致 |
| Judge 全失败 | 检查 `DEEPSEEK_API_KEY`；网络可否访问 `api.deepseek.com` |
| 大量 `Request timed out` | 增大 YAML `defaults.timeout`（如 1800）；失败题可断点续跑补 |
| 进度条 `judge_acc=0%` 只有本轮新题 | 正常；历史题不计入 live 显示，看 JSON 总数 |
| 中断后续跑 | 直接再跑同一命令；已完成 id 会跳过 |
| 想只测一个模型 | `--only MODEL_NAME` 或 YAML 里只留一个 |
| HF 数据集下载慢 | 本机若已有缓存会直接用；可设 `HF_TOKEN` |

---

## 9. 建议的交接检查清单

接手同学跑通下面几步即可视为交接完成：

1. [ ] `pip install -r requirements.txt` 成功  
2. [ ] `export OPENAI_API_KEY` / `DEEPSEEK_API_KEY`  
3. [ ] `curl` 通目标模型 `/v1/models`  
4. [ ] 复制 `example.yaml`，改好 `name` / `base_url`  
5. [ ] `./run_hle.sh ... --dry-run` 路径正确  
6. [ ] `./run_hle.sh ... --max_samples 3` 写出 JSON，且含 `response` + `judge_response`  
7. [ ] 会用「正确/已做」统计准确率  

---

## 10. 联系与参考

- 详细命令备忘：`docs/sglang_hle_commands.md`
- 快速上手：`hle_eval/README.md`
- 官方论文 / 榜单：https://lastexam.ai/
- 数据集：https://huggingface.co/datasets/cais/hle

如有服务侧 SGLang 启动参数问题（tp-size、reasoning-parser 等），需找对应推理机负责人；本仓库只负责**评测客户端**。
