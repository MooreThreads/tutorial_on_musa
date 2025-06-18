#!/bin/bash

set -e

# 默认参数配置
DEFAULT_PORT=8000
DEFAULT_HOST="localhost"
DEFAULT_GPU_UTIL=0.8
DEFAULT_TP_SIZE=1
DEFAULT_MODEL_LEN=4096
DEFAULT_TRUST_REMOTE_CODE=true
DEFUALT_SERVED_MODEL_NAME=""

CONTAINER_NAME=""
WEBUI="false"

declare -A PARAM_ALIASES=(
    ["-tp"]="--tensor-parallel-size"
    ["-pp"]="--pipeline-parallel-size"
    ["-p"]="--port"
)

MODEL=""
USER_ARGS=()

check_dependencies() {
    if ! command -v vllm >/dev/null 2>&1; then
        echo "错误: vLLM 未安装或不在 PATH 中" >&2
        exit 1
    fi
}

resolve_aliases() {
    local key="$1"
    if [[ -v PARAM_ALIASES[$key] ]]; then
        echo "${PARAM_ALIASES[$key]}"
    else
        echo "$key"
    fi
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -*)
                break
                ;;
            *)
                MODEL="$1"
                shift
                ;;
        esac
    done

    while [[ $# -gt 0 ]]; do
        param="$(resolve_aliases "$1")"
        case "$param" in
            --port)
                DEFAULT_PORT="$2"
                shift 2
                ;;
            --host)
                DEFAULT_HOST="$2"
                shift 2
                ;;
            --gpu-memory-utilization)
                DEFAULT_GPU_UTIL="$2"
                shift 2
                ;;
            --tensor-parallel-size|--tp-size)
                DEFAULT_TP_SIZE="$2"
                shift 2
                ;;
            --max-model-len)
                DEFAULT_MODEL_LEN="$2"
                shift 2
                ;;
            --trust-remote-code)
                DEFAULT_TRUST_REMOTE_CODE=true
                shift
                ;;
            --no-trust-remote-code)
                DEFAULT_TRUST_REMOTE_CODE=false
                shift
                ;;
            --served-model-name)
                DEFUALT_SERVED_MODEL_NAME="$2"
                shift 2
                ;;
            --container-name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --webui)
                WEBUI="true"
                shift
                ;;
            *)
                USER_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

validate_model_path() {
    if [ -z "$MODEL" ]; then
        echo "错误: 必须指定模型路径" >&2
        exit 1
    fi
    if [ ! -d "$MODEL" ]; then
        echo "错误: 模型目录不存在 - $MODEL" >&2
        exit 1
    fi
}

build_final_args() {
    FINAL_ARGS=(
        "$MODEL"
        "--port" "$DEFAULT_PORT"
        "--gpu-memory-utilization" "$DEFAULT_GPU_UTIL"
        "--tensor-parallel-size" "$DEFAULT_TP_SIZE"
        "--max-model-len" "$DEFAULT_MODEL_LEN"
        "--served-model-name" "${DEFUALT_SERVED_MODEL_NAME:-$(basename "$MODEL")}"
    )

    $DEFAULT_TRUST_REMOTE_CODE && FINAL_ARGS+=("--trust-remote-code")
    FINAL_ARGS+=("${USER_ARGS[@]}")
}


wait_for_log_update() {
    local log_file="$1"
    local server_pid="$2"
    local model_name="$3"
    local host="$4"
    local port="$5"

    # 设定超时时间（秒）
    local timeout=30
    local elapsed=0
    local no_change_count=0

    # 获取初始日志文件大小
    local last_size=$(stat -c%s "$log_file")

    while ((elapsed < timeout)); do
        local current_size=$(stat -c%s "$log_file")
	    local last_line=$(tail -n 5 "$log_file" 2>/dev/null | grep -E -v '^[[:space:]]*$')

        if grep -q -E "Application startup complete" <<< "$last_line" && \
	    [ "$current_size" -ne "$last_size" ]; then
            if [ "$WEBUI" == "true" ]; then
                echo -e "\e[32mInstalling gradio...\e[0m"  >&2
                pip install gradio
                echo -e "\e[32mStart gradio webui...\e[0m"  >&2
                echo -e "\e[32mContainer: $CONTAINER_NAME\e[0m"  >&2
                setsid python -u ./gradio_demo/app.py --ip "$host" --port "$port" --model-name "$model_name" | tee -a webui.log &
                wait $!  # 等待该进程结束
                exit 0
            else
                echo -e "\e[32m"
                if [ -z "$CONTAINER_NAME" ]; then
                    echo "Please send the following request to obtain the model inference result."
                else
                    echo "Please send the following request in container($CONTAINER_NAME) to obtain the model inference result."
                fi
                cat <<EOF


curl http://$host:$port/v1/chat/completions -H "Content-Type: application/json" -d '{
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


start_vllm() {

    echo "========================================"
    echo "Starting vLLM Service..."
    echo "Model path        : $MODEL"
    echo "Served Model Name : ${DEFUALT_SERVED_MODEL_NAME:-$(basename "$MODEL")}"
    echo "Port              : $DEFAULT_PORT"
    echo "GPU Utilization   : $DEFAULT_GPU_UTIL"
    echo "Tensor Parallel   : $DEFAULT_TP_SIZE"
    echo "Max Model Length  : $DEFAULT_MODEL_LEN"
    echo "Trust Remote Code : $DEFAULT_TRUST_REMOTE_CODE"
    echo "Extra Arguments   : ${USER_ARGS[*]}"
    echo "Full Command      : vllm serve ${FINAL_ARGS[*]}"
    echo "========================================"

    LOG_FILE="vllm_serve.log"
    : > "$LOG_FILE"  # 清空日志文件

    setsid bash -c "stdbuf -oL vllm serve ${FINAL_ARGS[*]} 2>&1 | tee -a $LOG_FILE" &


    wait_for_log_update $LOG_FILE $!  "${DEFUALT_SERVED_MODEL_NAME:-$(basename "$MODEL")}" "$DEFAULT_HOST" "$DEFAULT_PORT"
}




# 主流程
check_dependencies
parse_arguments "$@"
validate_model_path
build_final_args
start_vllm
