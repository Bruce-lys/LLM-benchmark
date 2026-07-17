# HLE SGLang Evaluation

用 YAML 配置 + shell 入口跑多模型 HLE 子集评测（含 realtime Judge）。

## 目录结构

```text
hle_eval/
├── run_hle.sh                 # 一键入口
├── README.md
├── configs/
│   ├── default.yaml           # 全局默认说明
│   ├── models/
│   │   ├── current.yaml       # 当前评测模型
│   │   └── example.yaml       # 换模型模板
│   └── legacy/                # 旧 JSON 配置（兼容）
├── src/                       # Python 代码
├── data/subsets/              # 评测子集
├── outputs/predictions/       # 模型预测结果 JSON
└── docs/                      # 详细命令说明
```

## 快速开始

```bash
cd /home/ubuntu/szy/hle
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=EMPTY
export DEEPSEEK_API_KEY=sk-xxx

cd hle_eval
./run_hle.sh configs/models/current.yaml
```

## 换模型

```bash
cp configs/models/example.yaml configs/models/my_model.yaml
# 编辑 name / base_url
./run_hle.sh configs/models/my_model.yaml
```

## 常用参数

```bash
./run_hle.sh configs/models/current.yaml --dry-run
./run_hle.sh configs/models/current.yaml --parallel 1
./run_hle.sh configs/models/current.yaml --only Ornith-0.9_Qwen3.6-0.1
./run_hle.sh configs/models/current.yaml --max_samples 3
```

## 交接文档

完整评测流程、脚本说明、评分口径与 FAQ，见：

**[`docs/HANDOVER.md`](docs/HANDOVER.md)**

命令备忘见 `docs/sglang_hle_commands.md`。
