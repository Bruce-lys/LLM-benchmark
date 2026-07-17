#!/usr/bin/env bash
# 用法: ./check_endpoint.sh http://10.0.3.52:8302/v1 Qwen3.6-35B-A3B
# 三步验证一个 vLLM 端点能否用于 SWE-bench: 连通 -> 模型名 -> 工具调用解析
set -e
EP="${1:?endpoint}"; NAME="${2:?served_model_name}"
echo "[1] 连通性+模型名:"
curl -sf -m 10 "$EP/models" | python3 -c "
import json,sys; ids=[m[\"id\"] for m in json.load(sys.stdin)[\"data\"]]
print(\"  served:\", ids); exit(0 if \"$NAME\" in ids else 1)" || { echo "  FAIL: 模型名不匹配或端点不通"; exit 1; }
echo "[2] 工具调用解析（SWE-bench 的命脉，配错 parser 分数崩 10-28pp）:"
curl -sf -m 120 "$EP/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\": \"$NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"List files using the bash tool.\"}],
  \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}, \"required\": [\"command\"]}}}],
  \"max_tokens\": 2048}" | python3 -c "
import json,sys
c=json.load(sys.stdin)[\"choices\"][0]
tc=c[\"message\"].get(\"tool_calls\")
assert tc and tc[0][\"function\"][\"name\"]==\"bash\", \"没有解析出 bash tool_call —— 检查 --tool-call-parser 是否匹配模型家族\"
print(\"  OK: finish=%s, tool_call=%s\" % (c[\"finish_reason\"], tc[0][\"function\"][\"arguments\"][:60]))"
echo "[3] 通过。下一步: 复制 configs/ 里的 yaml, 改 tag/served_name/endpoint, 先 slice 0:1 冒烟。"
