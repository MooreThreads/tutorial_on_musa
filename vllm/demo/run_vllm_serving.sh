#!/bin/bash
set -e

MODEL_NAME="$1"
CONFIG_FILE="model_config.json"

if [ -z "$MODEL_NAME" ]; then
    echo "× Please provide the model name, for example:"
    echo "   ./run_vllm_serving.sh DeepSeek-R1-Distill-Qwen-1.5B"
    exit 1
fi

# 用 Python 解析 JSON 获取 URL
MODEL_URL=$(python3 -c "
import json
config_file = '$CONFIG_FILE'
model_name = '$MODEL_NAME'
with open(config_file, 'r') as f:
    data = json.load(f)
print(data.get(model_name, ''))
")

if [ -z "$MODEL_URL" ]; then
    echo "× $MODEL_NAME is not supported yet, please refer to the website to try other models: https://docs.mthreads.com/mtt/mtt-doc-online/compability"
    exit 1
fi

echo "√ 找到模型 URL: $MODEL_URL"

# 目录和日志路径
CURRENT_DIR=$(pwd)
MODEL_DIR="/data/mtt/models"
CONVERTED_MODEL_DIR="/data/mtt/models_convert"
LOG_FILE="/data/mtt/logs/model_server.log"
MODEL_CHECK_FILE="$MODEL_DIR/$MODEL_NAME/model.safetensors"  
SUCCESS_MESSAGE="INFO:     Started server process"

# 确保目录存在
mkdir -p "$MODEL_DIR" "$CONVERTED_MODEL_DIR" "$(dirname "$LOG_FILE")"

# 检查模型是否已经存在
if [ -f "$MODEL_CHECK_FILE" ]; then
    echo "√ The model file already exists. Skip the download step."
else
    echo "⬇ The model file does not exist, start downloading the model..."
    cd "$MODEL_DIR"
    apt update && apt install -y git-lfs jq
    git lfs install
    git clone "$MODEL_URL" "$MODEL_NAME"
    echo "√ Model download completed."
fi

# 权重转换
cd "${CURRENT_DIR}/.."
./convert_weight.sh "$MODEL_DIR/$MODEL_NAME" 1

# 启动 vLLM 服务器
python -m vllm.entrypoints.openai.api_server \
        --model "$CONVERTED_MODEL_DIR/$MODEL_NAME-tp1-convert" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        -pp 1 \
        --block-size 64 \
        --max-model-len 2048 \
        --disable-log-stats \
        --disable-log-requests \
        --device "musa" \
        --served-model-name model-develop_test > "$LOG_FILE" 2>&1 &

pid=$!
echo "Wait for the service to start..."
while true; do
    if grep -q "$SUCCESS_MESSAGE" "$LOG_FILE"; then
        echo "√ Service has been started. If it does not work, check the log: $LOG_FILE"
        break
    else
        echo "Wait for the service to start..."
        sleep 5  # 每隔 5 秒检查日志文件
    fi
done
