#!/bin/bash

# 输入矩阵大小列表
input_data=$(cat <<'EOF'
128 128 128
256 256 256
512 512 512
1024 1024 1024
2048 2048 2048
4096 4096 4096
8192 8192 8192
4098 4098 4098
8190 8190 8190
8192 768 8192
EOF
)

# 每组测试迭代次数
test_iter=1000

# 测试类型列表
TEST_TYPES=("fp64" "tf32")

# GEMM 可执行文件目录
EXE_DIR="./fp64_tf32_src"

# 日志目录
LOG_DIR="mudnn_bench_logs"
mkdir -p "$LOG_DIR"
ABS_LOG_DIR=$(realpath "$LOG_DIR")
log_file="${ABS_LOG_DIR}/bench_fp64_tf32_types.log"
> "$log_file"

# Python 分析脚本路径
PYTHON_SUMMARIZE="exetrct_log_tools/summarize_fp64_tf32_log.py"

for type in "${TEST_TYPES[@]}"; do
    echo "=============================="
    echo "开始测试：$type"
    echo "=============================="

    # 根据类型选择可执行文件
    if [[ "$type" == "fp64" ]]; then
        exe="${EXE_DIR}/gemm_fp64"
    elif [[ "$type" == "tf32" ]]; then
        exe="${EXE_DIR}/gemm_tf32"
    else
        echo "未知类型: $type"
        continue
    fi

    # 检查可执行文件是否存在
    if [[ ! -f "$exe" ]]; then
        echo "错误：找不到可执行文件 $exe"
        continue
    fi

    # 遍历矩阵大小
    while read -r m n k; do
        # 清理可能的空格
        m=$(echo "$m" | tr -d ' ')
        n=$(echo "$n" | tr -d ' ')
        k=$(echo "$k" | tr -d ' ')

        echo "矩阵大小: M=$m, N=$n, K=$k"

        if [[ -n "$m" && -n "$n" && -n "$k" ]]; then
            # 执行 GEMM 测试并记录日志
            MUSA_VISIBLE_DEVICES=7 "$exe" "$m" "$n" "$k" "$test_iter" >> "$log_file" 2>&1
            sleep 1
        fi
    done <<< "$input_data"

done

# 调用 Python 分析脚本
if [[ -f "$PYTHON_SUMMARIZE" ]]; then
    python "$PYTHON_SUMMARIZE" "$log_file"
else
    echo "警告：Python 分析脚本不存在: $PYTHON_SUMMARIZE"
fi

echo "所有测试完成！日志目录：$ABS_LOG_DIR"

