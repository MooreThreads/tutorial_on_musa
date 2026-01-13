import re
import sys
import csv
import os

if len(sys.argv) < 2:
    print("Usage: python summary_fix_data.py <log_file>")
    sys.exit(1)

log_file = sys.argv[1]
print(f"📊 正在读取并解析日志：{log_file}")

if not os.path.exists(log_file):
    print("❌ 日志文件不存在")
    sys.exit(1)

# 收集结果
records = []

# 正则模式
re_start = re.compile(r"测试:\s*M=(\d+),\s*N=(\d+),\s*K=(\d+),\s*Type=([\w:]+)")
re_result = re.compile(r"AverageElapsedTime\(ms\)\s*:\s*([\d\.]+)\s*,\s*Throughput\s*([\d\.]+)\s*GOPS")

cur_M = cur_N = cur_K = cur_type = None

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # 匹配开始参数
        m1 = re_start.search(line)
        if m1:
            cur_M, cur_N, cur_K, cur_type = m1.groups()
            continue

        # 匹配结果
        m2 = re_result.search(line)
        if m2 and cur_M is not None:
            elapsed, gops = m2.groups()
            records.append({
                "M": cur_M,
                "N": cur_N,
                "K": cur_K,
                "Type": cur_type,
                "AvgTime(ms)": elapsed,
                "GOPS": gops
            })
            # 清空当前块（防止串行）
            cur_M = cur_N = cur_K = cur_type = None

# 输出 CSV
if not records:
    print("⚠️ 未提取到任何有效数据")
    sys.exit(0)

csv_path = log_file.replace(".log", ".csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"✅ 解析完成，共 {len(records)} 条数据")
print(f"📄 CSV 已生成：{csv_path}")

