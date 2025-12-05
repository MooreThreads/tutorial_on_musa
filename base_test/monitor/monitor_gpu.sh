#!/bin/bash

# 默认参数
device_id=0
interval=1
count=-1
output_file="gpu_monitor_log.txt"

# ANSI 颜色
RED='\033[0;31m'
NC='\033[0m' # 清除颜色

# 报警计数器
high_temp_count=0
low_freq_count=0

# 解析命令行参数（不包括 -o）
while getopts "d:i:n:" opt; do
  case $opt in
    d) device_id=$OPTARG ;;
    i) interval=$OPTARG ;;
    n) count=$OPTARG ;;
    *) echo "Usage: $0 [-d device_id] [-i interval_sec] [-n count]"
       exit 1 ;;
  esac
done

# 写入启动信息（仅首次）
if [ ! -f "$output_file" ]; then
    echo "=== GPU Monitor Started at $(date) ===" >> "$output_file"
    echo "Device: $device_id, Interval: $interval sec" >> "$output_file"
    echo "----------------------------------------" >> "$output_file"
fi

# 开始监控
i=0
while [ $count -lt 0 ] || [ $i -lt $count ]; do
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    temp_line=$(mthreads-gmi -i "$device_id" -q | grep "GPU Current Temp")
    freq_line=$(mthreads-gmi -i "$device_id" -q | grep "Graphics")

    temp=$(echo "$temp_line" | grep -oP '\d+(?=C)')
    freq=$(echo "$freq_line" | grep -oP '\d+(?=MHz)')

    plain_msg=""
    color_msg=""

    # 检查高温
    if [ "$temp" -gt 95 ]; then
        plain_msg+="高温报警(${temp}C)"
        color_msg+="${RED}高温报警(${temp}C)${NC} "
        ((high_temp_count++))
    fi

    # 检查降频
    if [ "$freq" -lt 1750 ]; then
        plain_msg+=" 降频警告(${freq}MHz)"
        color_msg+="${RED}降频警告(${freq}MHz)${NC}"
        ((low_freq_count++))
    fi

    {
        echo "[$timestamp]"
        echo "$temp_line"
        echo "$freq_line"
        [ -n "$plain_msg" ] && echo "$plain_msg"
        echo "------------------------------------"
    } >> "$output_file"

    # 终端打印报警（红色）
    [ -n "$plain_msg" ] && echo -e "[$timestamp] $color_msg"

    sleep "$interval"
    ((i++))
done

# 打印统计结果
echo -e "\n=== 监控结束 ==="
echo "高温次数: $high_temp_count"
echo "降频次数: $low_freq_count"

