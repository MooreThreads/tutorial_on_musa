import re
import os
import sys
from typing import List, Dict, Optional

def extract_matmul_data(log_path: str) -> List[Dict[str, str]]:
    patterns = {
        "datatype": re.compile(r"DataType (\w+)"),
        "mat_params": re.compile(r"m (\d+), n (\d+), k (\d+)"),
        "elapsed_time": re.compile(r"AverageElapsedTime\(ms\) : (\d+\.\d+)"),
        "throughput_gops": re.compile(r"Throughput (\d+\.\d+) GOPS")
    }

    extracted = []
    current_block = {}

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                dt_match = patterns["datatype"].search(line)
                if dt_match:
                    current_block["datatype"] = dt_match.group(1)

                mp_match = patterns["mat_params"].search(line)
                if mp_match:
                    current_block["m"] = mp_match.group(1)
                    current_block["n"] = mp_match.group(2)
                    current_block["k"] = mp_match.group(3)

                et_match = patterns["elapsed_time"].search(line)
                if et_match:
                    current_block["elapsed_time"] = et_match.group(1)

                tp_match = patterns["throughput_gops"].search(line)
                if tp_match:
                    tops = round(float(tp_match.group(1)) / 1000, 4)
                    current_block["throughput_tops"] = str(tops)

                if line == "==============================" and current_block:
                    required = ["datatype", "m", "n", "k", "elapsed_time", "throughput_tops"]
                    if all(key in current_block for key in required):
                        dim = f"{current_block['m']}-{current_block['n']}-{current_block['k']}"
                        extracted.append({
                            "datatype": current_block["datatype"],
                            "shape": dim,
                            "Throughput(TOPS)": current_block["throughput_tops"],
                            "AverageElapsedTime(ms)": current_block["elapsed_time"]
                        })
                    current_block = {}

        required = ["datatype", "m", "n", "k", "elapsed_time", "throughput_tops"]
        if current_block and all(key in current_block for key in required):
            dim = f"{current_block['m']}×{current_block['n']}×{current_block['k']}"
            extracted.append({
                "datatype": current_block["datatype"],
                "shape": dim,
                "Throughput(TOPS)": current_block["throughput_tops"],
                "AverageElapsedTime(ms)": current_block["elapsed_time"]
            })

    except Exception as e:
        print(f"❌ 读取日志失败：{str(e)}")
        return []

    return extracted

def generate_csv(data: List[Dict[str, str]], output_path: str) -> bool:
    if not data:
        print("⚠️  未提取到有效数据，跳过CSV生成")
        return False

    headers = ["datatype", "shape", "Throughput(TOPS)", "AverageElapsedTime(ms)"]

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(headers) + "\n")
            for item in data:
                row = [item[h] for h in headers]
                f.write(", ".join(row) + "\n")
        print(f"✅ CSV生成成功：{output_path}")
        return True
    except Exception as e:
        print(f"❌ 生成CSV失败：{str(e)}")
        return False

def main(input_log: str, output_csv: Optional[str] = None):
    if not os.path.isfile(input_log):
        print(f"❌ 输入日志文件不存在：{input_log}")
        return

    if not output_csv:
        log_dir = os.path.dirname(input_log)
        log_name = os.path.splitext(os.path.basename(input_log))[0]
        output_csv = os.path.join(log_dir, f"{log_name}_summary.csv")

    print(f"📊 开始提取日志数据：{input_log}")
    matmul_data = extract_matmul_data(input_log)

    if not matmul_data:
        print("❌ 未提取到任何有效测试数据")
        return

    print(f"✅ 成功提取 {len(matmul_data)} 条测试记录")

    generate_csv(matmul_data, output_csv)
    print("🎯 所有操作完成！")

if __name__ == "__main__":
    # 修正sys.argv判断（sys.argv[0]是脚本名，需至少传入1个输入文件路径）
    if len(sys.argv) < 2:
        print("用法：")
        print("  python summarize_fp64_tf32_log.py <输入日志文件路径>")
        print("示例：")
        print("  python summarize_fp64_tf32_log.py bench.log")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[1].replace('.log', '.csv')  # 日志文件同名CSV输出
    main(input_path, output_path)

