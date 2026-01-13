#!/bin/bash

set -e

input_data=$(cat <<'EOF'
128     128     128
256     256     256
512     512     512
1024    1024    1024
2048    2048    2048
4096    4096    4096
8192    8192    8192
4098    4098    4098
8190    8190    8190
8192    768     8192
EOF
)
test_iter=1000

TEST_TYPES=("f16:f16:f32:f32" "bf16:bf16:f32:f32" "f32" "int8" "q8:q8:f32:f32" "bf16:q4:bf16:bf16" "float8_e4m3:float8_e4m3:f16:f16")
# TEST_TYPES=("f32")
LOG_DIR="mudnn_bench_logs"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/bench_fix_matmul.log"
> "$log_file"

# 先测试命令是否存在
if [ ! -f "../bin/mudnn_bench" ]; then
    echo "错误：未找到 ../bin/mudnn_bench 可执行文件" | tee -a "$log_file"
    exit 1
fi

echo "开始测试，日志文件：$log_file"

for type in "${TEST_TYPES[@]}"; do
    echo "开始测试数据类型：$type" | tee -a "$log_file"
    
    # 使用 while 循环逐行读取
    echo "$input_data" | while IFS= read -r line; do
        # 跳过空行
        [ -z "$line" ] && continue
        
        # 使用 awk 或直接读取三个数字
        # 方法1：使用 read
        read m n k <<< "$line"
        
        # 或者方法2：使用 awk（更可靠）
        # m=$(echo "$line" | awk '{print $1}')
        # n=$(echo "$line" | awk '{print $2}')
        # k=$(echo "$line" | awk '{print $3}')
        
        echo "测试: M=$m, N=$n, K=$k, Type=$type" | tee -a "$log_file"
        
        # 检查参数是否正确
        if ! [[ "$m" =~ ^[0-9]+$ ]] || ! [[ "$n" =~ ^[0-9]+$ ]] || ! [[ "$k" =~ ^[0-9]+$ ]]; then
            echo "错误：参数不是数字: m=$m, n=$n, k=$k" | tee -a "$log_file"
            continue
        fi

        # 临时保存命令
        cmd="MUSA_VISIBLE_DEVICES=7 ../bin/mudnn_bench -m --mm_m=\"$m\" --mm_n=\"$n\" --mm_k=\"$k\" --warmup 30 --tm i --tmv \"$test_iter\" -p -c -t \"$type\""
        echo "执行命令: $cmd" >> "$log_file"
        
        # 执行命令并捕获退出状态
        if MUSA_VISIBLE_DEVICES=7 ../bin/mudnn_bench -m \
            --mm_m="$m" --mm_n="$n" --mm_k="$k" \
            --warmup 30 \
            --tm i \
            --tmv "$test_iter" \
            -p \
            -c \
            -t "$type" >> "$log_file" 2>&1; then
            echo "测试成功: M=$m, N=$n, K=$k, Type=$type" | tee -a "$log_file"
        else
            exit_code=$?
            echo "测试失败: M=$m, N=$n, K=$k, Type=$type, 退出码: $exit_code" | tee -a "$log_file"
        fi
        
        echo "----------------------------------------" >> "$log_file"
        sleep 2
    done
done

python sexetrct_log_tool/summary_mixed_data.py  "$log_file"
echo "所有测试完成！日志目录：$LOG_DIR"
echo "查看日志：cat $log_file"
