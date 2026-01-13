#!/bin/bash

input_data=$(cat <<'EOF'
128	128	128
256	256	256
512	512	512
1024	1024	1024
2048	2048	2048
4096	4096	4096
8192	8192	8192
4098	4098	4098
8190	8190	8190
EOF
)
test_iter=1000

TEST_TYPES=("f32" "f16" "bf16" "q8" "float8_e4m3" "float8_e5m2")
# TEST_TYPES=("f32")
LOG_DIR="mudnn_bench_logs"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/bench_f32_f16_bf16_q8_fp8.log"
> "$log_file"

for type in "${TEST_TYPES[@]}"; do
    echo "开始测试数据类型：$type"
    while IFS=$'\t' read -r m n k; do
        m=$(echo "$m" | tr -d ' ')
        n=$(echo "$n" | tr -d ' ')
        k=$(echo "$k" | tr -d ' ')
        echo "$m $n $k"
        
        if [[ -n "$m" && -n "$n" && -n "$k" ]]; then
            MUSA_VISIBLE_DEVICES=7 ../bin/mudnn_bench -m \
                -t "$type" \
                --mm_m="$m" --mm_n="$n" --mm_k="$k" \
                --mm_mode=0 \
                --tm i \
                --tmv "$test_iter" \
                -p \
                >> "$log_file" 2>&1 
            sleep 2
        fi
    done < <(echo "$input_data") 
done 

python exetrct_log_tools/summarize_f32_f16_bf16_q8_fp8_log.py "$log_file"

echo "所有测试完成！日志目录：$LOG_DIR"
