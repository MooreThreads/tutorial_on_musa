#!/bin/bash

# 默认值
MODEL=""
CONVERTED_MODEL=""
MODEL_TYPE=""  # 
TP_SIZE=""
MODEL_CONFIG_FILE="supported_models.json"
TIME=$(date "+%Y%m%d_%H%M%S")
DEFAULT_MODEL_DIR="/data/musa_develop_demo_$TIME"
DOWNLOAD_MODEL_DIR=""
vLLM_HOST=""
vLLM_PORT=""

# 解析参数的函数
parse_args() {
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)  # model_name
            TASK="$2"
            shift 2
            ;;
        --model)  # original model path
            MODEL="$2"
            shift 2
            ;;
        --converted-model)  # converted model path
            CONVERTED_MODEL="$2"
            shift 2
            ;;
        -tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --download-model-dir)
            DOWNLOAD_MODEL_DIR="$2"
            shift 2
            ;;
        --vllm-host)
            vLLM_HOST="$2"
            shift 2
            ;;
        --vllm-port)
            vLLM_PORT="$2"
            shift 2
            ;;
        --webui)
            WEBUI=true  # 如果传入了--webui参数，则设置为true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

WEBUI=${WEBUI:-false}

validate_args() {
if [[ -z "$TASK" ]]; then
    # 隐藏--container-name参数
    echo "Usage: $0 --task <model_name> [--model <original_model_path> | --converted-model <converted_model_path>] [-tp-size <tensor_parallel_size>]"
    exit 1
fi
}

validate_args
}


fetch_model_info() {
    local py_output
    local py_status
    local model_name="$1"
    local tp_size="$2"

    py_output=$(python3 -c "
import json
import sys

config_file = '$MODEL_CONFIG_FILE'
model_name = '$model_name'
tp_size = '$tp_size'

try:
    with open(config_file, 'r') as f:
        data = json.load(f)
except Exception as e:
    sys.stderr.write(f'× 配置文件读取失败: {str(e)}\\n')
    exit(1)

info = data.get(model_name, {})
if not info:
    sys.stderr.write(f'× 模型 \"{model_name}\" 不在支持列表中\\n')
    sys.stderr.write(f'✓ 支持的模型: {list(data.keys())}\\n')
    exit(2)

# 处理tensor_parallel_size
supported_tp_sizes = info.get('tensor_parallel_size', [])
if not isinstance(supported_tp_sizes, list):
    supported_tp_sizes = [supported_tp_sizes] if supported_tp_sizes else []

final_tp_size = None
if tp_size and tp_size.isdigit():
    if int(tp_size) not in supported_tp_sizes:
        sys.stderr.write(f'× 不支持的tensor_parallel_size值: {tp_size}\\n')
        sys.stderr.write(f'✓ 支持的TP大小: {supported_tp_sizes}\\n')
        exit(3)
    final_tp_size = tp_size
else:
    final_tp_size = supported_tp_sizes[-1] if supported_tp_sizes else 1

# 输出结果（用制表符分隔）
print('\t'.join([
    info.get('modelscope_url', ''),
    info.get('huggingface_url', ''),
    str(final_tp_size)
]))
" 2>&1)

    py_status=$?
    # 处理Python脚本错误
    if [ $py_status -ne 0 ]; then
        echo "$py_output" >&2
        exit $py_status
    fi

    echo "$py_output"
}

check_and_prepare_model() {

    if [ -z "$1" ]; then
        echo "错误：必须提供 model_name 参数" >&2
        return 1
    fi

    local model_name="$1"
    local model_path="$2"
    local converted_model_path="$3"
    local tp_size="$4"
    local model_url="$5"

    # 1. 如果只有 model_name, 下载模型
    if [ -z "$model_path" ] && [ -z "$converted_model_path" ]; then

        if [ -z "$DOWNLOAD_MODEL_DIR" ]; then
            model_path=$DEFAULT_MODEL_DIR/$model_name
            converted_model_path=$DEFAULT_MODEL_DIR/$model_name-tp$tp_size-converted

            mkdir -p "$model_path" "$converted_model_path"
        else
            model_path=$DOWNLOAD_MODEL_DIR/$model_name
            converted_model_path=$DOWNLOAD_MODEL_DIR/$model_name-tp$tp_size-converted
        fi

        echo -e "\e[32mmodel_path: $model_path\e[0m" >&2
        echo -e "\e[32mconverted_model_path: $converted_model_path\e[0m" >&2
        apt-get update -qq >&2 && apt-get install -y --no-install-recommends git-lfs jq >&2
        git lfs install >&2
        git clone "$model_url" "$model_path" >&2  # TODO(wangkang): need check for clone if successful
        echo "√ Model download completed." >&2


    # 2. 如果只有 model_path 没有 converted_model_path
    elif [ -n "$model_path" ] && [ -z "$converted_model_path" ]; then

        # not found model dir
        if [ ! -e "$model_path" ]; then
            echo "Erro: Not found model path $model_path" >&2
            exit 1
        fi
        converted_model_path=$(dirname "$model_path")/$model_name-tp$tp_size-converted
        echo "Automatically generate converted_model_path: $converted_model_path" >&2
        mkdir -p "$converted_model_path"
    
    elif [ -n "$converted_model_path" ] && [ ! -e "$converted_model_path" ]; then
        echo "Erro: Not found converted model path $model_path" >&2
        exit 1

    fi

    if [ -z "$(ls -A $converted_model_path)" ]; then
        convert_weight $model_path $converted_model_path $tp_size $MODEL_TYPE >&2
    fi

    echo "$converted_model_path"
}


convert_weight() {
    local model_dir="$1"
    local converted_model_dir="$2"
    local tp_size="$3"
    local model_type="$4"

    local python_cmd=(
        python -u  # -u 参数禁用缓冲
        -m mttransformer.convert_weight
        --in_file "${model_dir}"
        --saved_dir "${converted_model_dir}"
        --tensor-para-size "${tp_size}"
    )
    
    [[ -n "${model_type}" ]] && python_cmd+=(--model-type "${model_type}")

    "${python_cmd[@]}"
}


wait_for_log_update() {
    local log_file="$1"
    local server_pid="$2"
    local model_name="$3"
    local model_path="$4"

    # 设定超时时间（秒）
    local timeout=30
    local elapsed=0
    local no_change_count=0

    # 获取初始日志文件大小
    local last_size=$(stat -c%s "$log_file")

    while ((elapsed < timeout)); do
        local current_size=$(stat -c%s "$log_file")
	    local last_line=$(tail -n 5 "$log_file" 2>/dev/null | grep -E -v '^[[:space:]]*$')

        if grep -q -E "Uvicorn running on http://" <<< "$last_line" && \
	    [ "$current_size" -ne "$last_size" ]; then
            if [ "$WEBUI" == "true" ]; then
                echo -e "\e[32m"
                echo "Start gradio webui..."
                pip install gradio
                create_web_ui $host $port $model_name
                echo -e "\e[0m"
            else
                echo -e "\e[32m"
                if [ -z "$CONTAINER_NAME" ]; then
                    echo "Please send the following request to obtain the model inference result."
                else
                    echo "Please send the following request in container($CONTAINER_NAME) to obtain the model inference result."
                fi
                cat <<EOF


curl http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "$model_name",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the NBA final series in 2020?"}
    ]
}'
EOF
                echo -e "\e[0m"
                break
            fi

        fi

        last_size=$current_size
    done

    while ((elapsed < timeout)); do
        sleep 30
        ((elapsed += 2))

        # 更新上次记录的大小
        last_size=$current_size

        # 检查进程是否仍在运行
        if ! ps -p "$server_pid" > /dev/null; then
            echo "Error: vLLM server process exited unexpectedly. Check logs: $log_file"
            return 1
        fi
    done
    return 1
}

create_web_ui() {
    local ip="$1"
    local port="$2"
    local served_model_name="$3"

    python ./gradio_demo/app.py --ip "$ip" --port "$port" --model-name "$served_model_name"
}

start_server() {
    # 解析传入的参数
    local converted_model_path="$1"
    local tensor_parallel_size="$2"
    local served_model_name="$3"
    local host="$4"
    local port="$5"
    
    log_file=$(dirname "$converted_model_path")/model_server.log
    : > "$log_file"
    echo "Wait for the service to start..."
    # 初始化命令
    cmd=(
        setsid
        python -m vllm.entrypoints.openai.api_server
        --model "$converted_model_path"
        --trust-remote-code
        --tensor-parallel-size "$tensor_parallel_size"
        -pp 1
        --block-size 64
        --max-model-len 2048
        --disable-log-stats
        --disable-log-requests
        --device "musa"
        --served-model-name "$served_model_name"
    )

    [[ -n "$host" ]] && cmd+=(--host "$host")
    [[ -n "$port" ]] && cmd+=(--port "$port")

    # 执行命令
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee -a "$log_file" &


    SERVER_PID=$!

    wait_for_log_update "$log_file" "$SERVER_PID" "$served_model_name" "$converted_model_path" "$host" "$port"
}



# 主函数
main() {
  parse_args "$@"

  # load json
  if ! output=$(fetch_model_info "$TASK" "$TP_SIZE"); then
    exit $?
  fi
  read -r ms_url hf_url tp_size <<< "$output"

  # prepare model
  if ! output=$(check_and_prepare_model "$TASK" "$MODEL" "$CONVERTED_MODEL" "$tp_size" "$ms_url"); then
    exit $?
  fi
  read -r converted_model_path <<< "$output"

  start_server "$converted_model_path" "$tp_size" "$TASK" "$vLLM_HOST" "$vLLM_PORT"

}

# 执行主函数
main "$@"