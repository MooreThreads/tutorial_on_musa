#!/bin/bash

set -e

# 默认参数配置
DEFAULT_PORT=8000
DEFAULT_GPU_UTIL=0.8
DEFAULT_TP_SIZE=1
DEFAULT_MODEL_LEN=4096
DEFAULT_TRUST_REMOTE_CODE=true

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
    )

    $DEFAULT_TRUST_REMOTE_CODE && FINAL_ARGS+=("--trust-remote-code")
    FINAL_ARGS+=("${USER_ARGS[@]}")
}

start_vllm() {
    echo "========================================"
    echo "启动 vLLM 服务"
    echo "模型路径: $MODEL"
    echo "端口: $DEFAULT_PORT"
    echo "GPU 内存利用率: $DEFAULT_GPU_UTIL"
    echo "张量并行度: $DEFAULT_TP_SIZE"
    echo "最大模型长度: $DEFAULT_MODEL_LEN"
    echo "信任远程代码: $DEFAULT_TRUST_REMOTE_CODE"
    echo "其他参数: ${USER_ARGS[*]}"
    echo "完整命令: vllm serve ${FINAL_ARGS[*]}"
    echo "========================================"
    exec vllm serve "${FINAL_ARGS[@]}"
}

# 主流程
check_dependencies
parse_arguments "$@"
validate_model_path
build_final_args
start_vllm
