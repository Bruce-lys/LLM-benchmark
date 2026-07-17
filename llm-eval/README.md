# LLM Eval

|| [Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b) (official / ours) | [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b) (official / ours) |
| :---: | :---: | :---: |
| [AIME26](https://huggingface.co/datasets/MathArena/aime_2026) | 94.1 / 93.3 | 92.7 / - |
| [GPQA Diamond](https://huggingface.co/datasets/fingertap/GPQA-Diamond) | 87.8 / 86.4 | 86.0 / - |
| [NL2Repo](https://github.com/multimodal-art-projection/NL2RepoBench) | 36.2 / - | 29.4 / - |

需要独立沙箱环境的基准评估均作为独立项目在 `sandbox` 子目录维护：

- [x] NL2Repo: [sandbox/nl2repo-bench/README.md](sandbox/nl2repo-bench/README.md)

无需独立沙箱环境的基准评估在根目录维护：

- [x] AIME26
- [x] GPQA Diamond

## 环境配置

同步环境：

```bash
uv sync
source .venv/bin/activate
```

下载数据集：

```bash
hf download MathArena/aime_2026 \
  --repo-type=dataset \
  --local-dir data/aime26 \
  --max-workers 4

hf download fingertap/GPQA-Diamond \
  --repo-type=dataset \
  --local-dir data/gpqa_diamond \
  --max-workers 4
```

检测你的模型服务是否正常：

```bash
# IP 和端口改成你的模型服务
curl http://127.0.0.1:8000/v1/models | jq
```

## 模型配置

下载模型：

```bash
MODEL_PATH="/cpfs01/models/Qwen3.6-27B"
hf download Qwen/Qwen3.6-27B --local-dir "$MODEL_PATH"
```

启动 SGLang：

```bash
docker run -d \
  --name sglang-local \
  --runtime nvidia \
  --gpus '"device=0,1,2,3"' \
  --platform linux/arm64 \
  -v /cpfs01/models:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  lmsysorg/sglang:v0.5.11-cu130 \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key sk-vincent \
    --tp-size 4 \
    --context-length 262144 \
    --allow-auto-truncate \
    --dtype bfloat16 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

检查推理服务：

```bash
# 检查模型列表
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer sk-vincent" | jq

# 简单请求
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-vincent" \
  -H "Content-Type: application/json" \
  -d '{
       "model": "Qwen3.6-35B-A3B",
       "messages": [
          {"role": "system", "content": "Reply in Chinese."},
          {"role": "user", "content": "Introduce yourself briefly"}
        ]
      }' | jq
```

## 开始评估

所有的结果保存在 `output/<time>_<benchmark>_<model>` 文件夹。

### AIME26

```bash
python src/main.py \
  --benchmark aime26 \
  --model Qwen3.6-27B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 81920 \
  --num-workers 1 \
  --seed 42
```

### GPQA Diamond

```bash
python src/main.py \
  --benchmark gpqa_diamond \
  --model Qwen3.6-27B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 81920 \
  --num-workers 1 \
  --seed 42
```

## 断点续评

如果需要复用运行，删除对应 log dir 中 predictions.jsonl 中的无效行后，添加两个新的运行参数后重新运行即可，所有 benchmark 均支持。示例命令：

```bash
python src/main.py \
  --benchmark aime26 \
  --output-dir outputs/2026-07-02_22-32-25_aime26_Qwen-Qwen3.6-27B \
  --resume
```


---

Source: https://github.com/Explorer-Dong/llm-eval
