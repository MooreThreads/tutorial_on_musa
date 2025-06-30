#!/bin/bash
# 最好保证两台机器ssh免密

# 配置机器信息,请按照实际修改
SERVER_A="A_IP"
SERVER_B="B_IP"
USER="actual_username"
IB_DEVICES=("mlx5_2" "mlx5_3" "mlx5_4" "mlx5_5" "mlx5_8" "mlx5_9" "mlx5_10" "mlx5_11")
FAIL_LOG="failures_$(date +%Y%m%d).log"  # 含日期的日志文件

# 颜色定义
RED='\033[1;31m'
GREEN='\033[1;32m'
NC='\033[0m'  # 重置颜色

# 失败组合记录数组
declare -a FAILED_PAIRS

cleanup() {
    ssh ${USER}@${SERVER_A} "pkill -f 'ib_write_bw -d'" >/dev/null 2>&1
    ssh ${USER}@${SERVER_B} "pkill -f 'ib_write_bw -d'" >/dev/null 2>&1
}
trap cleanup EXIT

# 测试结果处理函数
process_result() {
    local dev_a=$1
    local dev_b=$2
    local log_file="client_${dev_a}_${dev_b}.log"
    
    if grep -q "BW average" "$log_file"; then
        echo -e "${GREEN}[PASS]${NC} $dev_b -> $dev_a"
        grep -A 5 "BW average" "$log_file" | tail -6
    else
        echo -e "${RED}[FAIL]${NC} $dev_b -> $dev_a"
        FAILED_PAIRS+=("$dev_a-$dev_b")
        # 记录详细失败日志
        echo "===== 失败组合: $dev_a-$dev_b =====" >> "$FAIL_LOG"
        cat "$log_file" >> "$FAIL_LOG"
        echo -e "\n" >> "$FAIL_LOG"
    fi
}

# 主测试循环
for ((i=0; i<${#IB_DEVICES[@]}; i++)); do
    DEV_A="${IB_DEVICES[$i]}"
    echo "[INFO] 机器A启动持续接收服务: ${DEV_A}"

    ssh ${USER}@${SERVER_A} "while true; do ib_write_bw -d ${DEV_A}; done" > "server_${DEV_A}.log" &
    SERVER_PID=$!
    sleep 8

    for ((j=0; j<${#IB_DEVICES[@]}; j++)); do
        DEV_B="${IB_DEVICES[$j]}"
        echo "[TEST] 机器B使用设备: ${DEV_B} -> 机器A设备: ${DEV_A}"

        # 执行测试并捕获完整输出
        ssh ${USER}@${SERVER_B} "ib_write_bw -d ${DEV_B} ${SERVER_A} -D 5" > "client_${DEV_A}_${DEV_B}.log"
        process_result "$DEV_A" "$DEV_B"
        sleep 2
    done

    kill -15 $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    ssh ${USER}@${SERVER_A} "pkill -f 'ib_write_bw -d ${DEV_A}'" >/dev/null 2>&1
done

# 失败组合总结
summarize_failures() {
    if [ ${#FAILED_PAIRS[@]} -eq 0 ]; then
        echo -e "${GREEN}所有组合测试成功！${NC}"
        return
    fi

    echo -e "\n${RED}===== 失败组合总结 =====${NC}"
    echo "共 ${#FAILED_PAIRS[@]} 组失败:"
    for pair in "${FAILED_PAIRS[@]}"; do
        echo -e "${RED}  $pair${NC}"
    done
    
    # 记录到日志文件
    echo -e "\n===== $(date) 失败组合汇总 =====" >> "$FAIL_LOG"
    printf "%s\n" "${FAILED_PAIRS[@]}" >> "$FAIL_LOG"
    echo -e "\n详细日志见: ${RED}$FAIL_LOG${NC}"
}

summarize_failures