from FlagEmbedding import BGEM3FlagModel
import torch
import torch_musa
import time
import numpy as np
import concurrent.futures
import random
import string
from tqdm import tqdm

# 生成1024 tokens的长文本（约1500-1800字符）
def generate_long_text(target_tokens=1024):
    """生成符合目标token长度的随机文本"""
    words = []
    current_tokens = 0
    while current_tokens < target_tokens:
        word_len = random.randint(3, 10)
        words.append(''.join(random.choices(string.ascii_letters, k=word_len)))
        current_tokens += 1
    return " ".join(words)

def process_batch(model, batch_sentences, max_length):
    """处理单个批次并返回结果和耗时"""
    start_time = time.time()
    embeddings = model.encode(batch_sentences, max_length=max_length)['dense_vecs']
    end_time = time.time()
    return embeddings, end_time - start_time

def warmup_model(model, batch_size=32, max_length=512, iterations=10):
    """执行模型预热以消除冷启动影响"""
    warmup_sentences = ["Warmup sentence " * 20] * batch_size  # 模拟长文本
    for _ in range(iterations):
        model.encode(warmup_sentences, max_length=max_length)['dense_vecs']

if __name__ == '__main__':
    # 初始化模型
    model = BGEM3FlagModel('./bge-m3', use_fp16=True, device='musa:0')
    
    # ===== 关键优化1：生成1024 tokens的长文本 =====
    print("=== Generating 1024-token texts ===")
    long_query = generate_long_text(1024)
    long_passage = generate_long_text(1024)
    
    # ===== 关键优化2：添加长文本预热 =====
    print("\n=== Starting model warm-up with long texts ===")
    warmup_model(model, batch_size=32, max_length=1024, iterations=10)
    print("=== Warm-up completed ===\n")
    
    # ===== 准备30个批次的并行任务 =====
    batch_pairs_list = []
    for _ in range(30):
        batch = []
        for _ in range(32):  # 每个批次32个样本
            q = generate_long_text(1024) if random.random() > 0.5 else long_query
            p = generate_long_text(1024) if random.random() > 0.5 else long_passage
            batch.append(q)
            batch.append(p)
        batch_pairs_list.append(batch)
    
    # ===== 并行执行30个批次 =====
    print("=== Starting 30 parallel batch processing ===")
    total_tokens = 0
    for batch in batch_pairs_list:
        total_tokens += sum(len(text.split()) for text in batch)
    
    start_time = time.time()
    batch_results = []
    batch_times = []
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_batch, model, batch, 1024) for batch in batch_pairs_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            embeddings, batch_time = future.result()
            batch_results.append(embeddings)
            batch_times.append(batch_time)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # ===== 性能指标计算 =====
    # 1. 吞吐量指标
    throughput_batches = len(batch_results) / total_time
    throughput_tokens = total_tokens / total_time
    
    # 2. 延迟指标
    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)
    
    # ===== 性能报告 =====
    print("\n===== Performance Report =====")
    print(f"Total batches processed: {len(batch_results)}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("\n--- Throughput ---")
    print(f"Throughput (batches/sec): {throughput_batches:.2f}")
    print(f"Throughput (tokens/sec): {throughput_tokens:.2f}")
    print("\n--- Latency ---")
    print(f"Avg batch time: {avg_batch_time:.4f} sec")
    print(f"Max batch time: {max_batch_time:.4f} sec")
    print(f"Min batch time: {min_batch_time:.4f} sec")
    print("=============================")
    
    # 示例相似度计算
    embeddings_1 = batch_results[0][0:2]  # 取第一个批次的前两个查询
    embeddings_2 = batch_results[0][2:4]  # 取第一个批次的前两个段落
    similarity = np.dot(embeddings_1, embeddings_2.T)
    print(f"\nSample similarity matrix:\n{similarity}")
