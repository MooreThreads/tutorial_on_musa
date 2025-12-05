import re
import sys
import os
from typing import List, Dict, Optional

def extract_matmul_data(log_path: str) -> List[Dict[str, str]]:
    patterns = {
        "datatype": re.compile(r"MatMul (\w+) Test \(MUSA\)"),
        "mat_params": re.compile(r"m = (\d+), n = (\d+), k = (\d+)"),
        "duration_us": re.compile(r"Duration:(\s*[\d\.]+) us"),
        "tflops": re.compile(r"computation-\w+=(\s*[\d\.]+)")
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

                dur_match = patterns["duration_us"].search(line)
                if dur_match:
                    us_val = float(dur_match.group(1).strip())
                    ms_val = round(us_val / 1000, 6)
                    current_block["duration_ms"] = str(ms_val)

                tf_match = patterns["tflops"].search(line)
                if tf_match:
                    tf_val = tf_match.group(1).strip()
                    current_block["tflops"] = str(round(float(tf_val), 6))

                if line == "========================================" and current_block:
                    required = ["datatype", "m", "n", "k", "duration_ms", "tflops"]
                    if all(key in current_block for key in required):
                        shape = f"{current_block['m']}-{current_block['n']}-{current_block['k']}"
                        extracted.append({
                            "DataType": current_block["datatype"],
                            "shape": shape,
                            "Compute_ability(TFLOPS)": current_block["tflops"],
                            "AverageElapsedTime(ms)": current_block["duration_ms"]
                        })
                    current_block = {}

        required = ["datatype", "m", "n", "k", "duration_ms", "tflops"]
        if current_block and all(key in current_block for key in required):
            shape = f"{current_block['m']}-{current_block['n']}-{current_block['k']}"
            extracted.append({
                "DataType": current_block["datatype"],
                "shape": shape,
                "Compute_ability(TFLOPS)": current_block["tflops"],
                "AverageElapsedTime(ms)": current_block["duration_ms"]
            })

    except Exception as e:
        print(f"❌ 读取日志失败：{str(e)}")
        return []

    return extracted

def generate_csv(data: List[Dict[str, str]], output_path: str) -> bool:
    if not data:
        print("⚠️  未提取到有效数据，跳过CSV生成")
        return False

    headers = ["DataType", "shape", "Compute_ability(TFLOPS)", "AverageElapsedTime(ms)"]
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(headers) + "\n")
            for item in data:
                row = [item[h] for h in headers]
                f.write(",".join(row) + "\n")
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
    if len(sys.argv) < 2:
        print("用法：")
        print("  python summarize_fp64_tf32_log.py <输入日志文件路径>")
        print("示例：")
        print("  python summarize_fp64_tf32_log.py bench.log")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[1].replace('.log', '.csv')
    main(input_path, output_path)
