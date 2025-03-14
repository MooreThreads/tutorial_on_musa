#!/bin/bash
set -e

# 定义路径和参数
MODEL_DIR="/data/mtt/models"
CONVERTED_MODEL_DIR="/data/mtt/models_convert"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_URL="https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.git"
LOG_FILE="/data/mtt/logs/model_server.log"
MODEL_CHECK_FILE="$MODEL_DIR/$MODEL_NAME/model.safetensors"  # 检查的模型文件

# 确保目录存在
mkdir -p "$MODEL_DIR"
mkdir -p "$CONVERTED_MODEL_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 检查模型文件是否存在
if [ -f "$MODEL_CHECK_FILE" ]; then
    echo "模型文件已存在，跳过下载步骤。"
else
    echo "模型文件不存在，开始下载模型..."
    cd "$MODEL_DIR"
    git lfs install
    git clone "$MODEL_URL" "$MODEL_NAME"
    echo "模型下载完成。"
fi

# 转换权重
./convert_weight.sh "$MODEL_DIR/$MODEL_NAME" 1

# 启动服务
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
        --served-model-name deepseek_test > "$LOG_FILE" 2>&1 &

pid=$!
wait $pid

echo "服务已启动，日志文件位于: $LOG_FILE"