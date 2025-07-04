from FlagEmbedding import FlagReranker
import time
import numpy as np
import concurrent.futures
import random
import string

# 生成长文本（1024 tokens约1500-1800字符）
def generate_long_text(target_tokens=1024):
    words = []
    current_tokens = 0
    while current_tokens < target_tokens:
        word_len = random.randint(3, 10)
        words.append(''.join(random.choices(string.ascii_letters, k=word_len)))
        current_tokens += 1
    return " ".join(words)

def process_batch(reranker, batch_pairs):
    """处理单个批次并返回结果和耗时"""
    start_time = time.perf_counter()
    scores = reranker.compute_score(batch_pairs)
    end_time = time.perf_counter()
    return scores, end_time - start_time

def main():
    # 加载模型（FP16精度 + MUSA设备加速）
    reranker = FlagReranker('BAAI/bge-reranker-large', 
                           use_fp16=True, 
                           device="musa")

    # ===== 长文本优化：生成1024 tokens的输入 =====
    print("=== Generating 1024-token texts ===")
    long_query = generate_long_text(1024)
    long_passage = generate_long_text(1024)
    
    # ===== 关键优化：添加模型预热（使用长文本）===== [1,6](@ref)
    print("=== Starting model warm-up with long texts ===")
    warmup_pairs = [[long_query, long_passage]] * 16
    for _ in range(5):
        reranker.compute_score(warmup_pairs)
    print("=== Warm-up completed ===\n")

    # 单次长文本推理测试
    start_time = time.perf_counter()
    score = reranker.compute_score([long_query, long_passage])
    latency = (time.perf_counter() - start_time) * 1000
    print(f"Long text score: {str(score)} | Latency: {latency:.2f} ms")

    # 准备批量数据（30个并行任务）
    batch_pairs_list = []
    for _ in range(30):
        pairs = []
        for _ in range(64):  # 每个任务64个样本
            q = generate_long_text(1024) if random.random() > 0.5 else long_query
            p = generate_long_text(1024) if random.random() > 0.5 else long_passage
            pairs.append([q, p])
        batch_pairs_list.append(pairs)

    # ===== 并行执行30个任务 =====
    print("\n=== Starting 30 parallel batch processing ===")
    total_tokens = sum(
        sum(len(q.split()) + len(p.split()) for q, p in pairs) 
        for pairs in batch_pairs_list
    )
    batch_times = [] 
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 控制并发数
        futures = [executor.submit(process_batch, reranker, pairs) for pairs in batch_pairs_list]
        
        batch_results = []
        for future in concurrent.futures.as_completed(futures):
            scores, batch_time = future.result()
            batch_results.append(scores)
            batch_times.append(batch_time)
    
    total_time = time.perf_counter() - start_time

    # 性能统计
    total_pairs = 30 * 64  # 30任务 * 每任务64对
    throughput_pairs = total_pairs / total_time
    throughput_tokens = total_tokens / total_time

    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)

    print("\n===== Performance Report =====")
    print(f"Total batches processed: {len(batch_results)}")
    print(f"Total pairs processed: {total_pairs}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")

    print("\n--- Throughput ---")
    print(f"Throughput: {throughput_pairs:.2f} pairs/sec")
    print(f"Token throughput: {throughput_tokens:.2f} tokens/sec")
    
    print("\n--- Latency ---")
    print(f"Average batch time: {avg_batch_time:.4f} sec")
    print(f"Max batch time: {max_batch_time:.4f} sec")
    print(f"Min batch time: {min_batch_time:.4f} sec")

    print("=============================")



if __name__ == "__main__":
    main()
