# SGLang HLE Commands

完整交接说明（流程 / 脚本 / 评分口径）见 **[HANDOVER.md](HANDOVER.md)**。

以下命令默认在 `hle_eval/` 目录执行。

## 0. 推荐入口（YAML + shell）

```bash
cd /home/ubuntu/szy/hle
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="EMPTY"
export DEEPSEEK_API_KEY="<your-deepseek-key>"

cd hle_eval
cp configs/models/example.yaml configs/models/my_model.yaml
# 编辑 my_model.yaml 里的 name / base_url

./run_hle.sh configs/models/my_model.yaml
```

当前 3 模型：

```bash
./run_hle.sh configs/models/current.yaml
```

常用参数：

```bash
./run_hle.sh configs/models/current.yaml --dry-run
./run_hle.sh configs/models/current.yaml --parallel 1
./run_hle.sh configs/models/current.yaml --only Ornith-0.9_Qwen3.6-0.1
./run_hle.sh configs/models/current.yaml --max_samples 3
```

兼容旧 JSON：

```bash
./run_hle.sh configs/legacy/sglang_models.json --parallel 3
```

## 1. 准备环境

```bash
cd /home/ubuntu/szy/hle
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="EMPTY"
```

## 2. 检查 Thinking

```bash
cd /home/ubuntu/szy/hle/hle_eval
PYTHONPATH=src python src/check_sglang_thinking.py \
  --base_url http://10.0.3.50:8000/v1 \
  --model YOUR_MODEL_NAME
```

## 3. 生成子集

```bash
PYTHONPATH=src python src/create_hle_subset.py \
  --ratio 0.1 \
  --seed 42 \
  --stratify_by category,difficulty \
  --output data/subsets/hle_subset_10pct_category_difficulty_all.json
```

## 4. 批量评测（直接调 Python）

```bash
PYTHONPATH=src python src/run_sglang_batch.py \
  --config configs/models/current.yaml
```

## 5. 事后 Judge

```bash
PYTHONPATH=src python src/run_judge_results.py \
  --dataset cais/hle \
  --predictions outputs/predictions/hle_MODEL.json \
  --judge deepseek-v4-flash \
  --judge_provider compatible \
  --base_url https://api.deepseek.com \
  --api_key_env DEEPSEEK_API_KEY \
  --num_workers 20 \
  --subset_file data/subsets/hle_subset_10pct_category_difficulty_all.json \
  --output outputs/predictions/judged_hle_MODEL.json
```
