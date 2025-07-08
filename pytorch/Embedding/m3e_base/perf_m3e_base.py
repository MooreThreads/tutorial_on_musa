import time
import concurrent.futures
import numpy as np
import torch
import torch_musa
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 生成长文本（1024 tokens约1500-1800字符）
def generate_long_text(target_tokens=1024):
    """生成符合目标token长度的随机文本"""
    import random
    import string
    
    words = []
    current_tokens = 0
    while current_tokens < target_tokens:
        word_len = random.randint(3, 10)
        words.append(''.join(random.choices(string.ascii_letters, k=word_len)))
        current_tokens += 1
    return " ".join(words)

def process_batch(model, sentences, batch_size=32):
    """处理单个批次并返回结果和耗时"""
    start_time = time.perf_counter()
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    end_time = time.perf_counter()
    return embeddings, end_time - start_time

def count_tokens(texts):
    """近似计算token数量（实际应使用tokenizer）"""
    return sum(len(text.split()) for text in texts)

def main():
    # 初始化模型（使用MUSA设备加速）
    model = SentenceTransformer('moka-ai/m3e-base', device='musa')
    
    # ===== 长文本优化：生成1024 tokens的输入 =====
    print("=== Generating 1024-token texts ===")
    long_text = generate_long_text(1024)
    
    # ===== 关键优化：添加长文本模型预热 =====
    print("\n=== Starting model warm-up with long texts ===")
    warmup_sentences = [long_text] * 5  # 使用长文本预热
    for _ in range(10):
        model.encode(
            warmup_sentences,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    print("=== Warm-up completed ===\n")
    
    # ===== 准备30个并行任务 =====
    num_tasks = 30
    batch_size = 32
    batches = []
    
    # 生成30个批次的输入数据（每个批次包含batch_size个句子）
    for _ in range(num_tasks):
        sentences_batch = []
        for _ in range(batch_size):
            # 50%概率使用长文本，50%生成新文本
            if np.random.rand() > 0.5:
                sentences_batch.append(long_text)
            else:
                sentences_batch.append(generate_long_text(1024))
        batches.append(sentences_batch)
    
    # 计算总token数
    total_tokens = sum(count_tokens(batch) for batch in batches)
    total_sentences = num_tasks * batch_size
    
    # ===== 并行执行30个任务 =====
    print(f"=== Starting {num_tasks} parallel batch processing ===")
    start_time = time.perf_counter()
    
    # 记录GPU初始状态
    initial_mem = torch_musa.memory_allocated()
    
    # 使用线程池并行处理
    batch_results = []
    batch_times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_batch, model, batch, batch_size) 
                  for batch in batches]
        
        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc="Processing batches"):
            embeddings, batch_time = future.result()
            batch_results.append(embeddings)
            batch_times.append(batch_time)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # ===== 性能指标计算 =====
    # 1. 吞吐量指标
    throughput_batches = len(batch_results) / total_time
    throughput_sentences = total_sentences / total_time
    throughput_tokens = total_tokens / total_time
    
    # 2. 延迟指标
    avg_batch_time = sum(batch_times) / len(batch_times)
    max_batch_time = max(batch_times)
    min_batch_time = min(batch_times)
    
    
    # ===== 性能报告 =====
    print("\n===== Performance Report =====")
    print(f"Total batches processed: {len(batch_results)}")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    print("\n--- Throughput ---")
    print(f"Throughput (batches/sec): {throughput_batches:.2f}")
    print(f"Throughput (sentences/sec): {throughput_sentences:.2f}")
    print(f"Throughput (tokens/sec): {throughput_tokens:.2f}")
    
    print("\n--- Latency ---")
    print(f"Average batch time: {avg_batch_time:.4f} sec")
    print(f"Max batch time: {max_batch_time:.4f} sec")
    print(f"Min batch time: {min_batch_time:.4f} sec")
    
    print("=============================")
    
    # 打印第一个批次的第一个句子嵌入示例
    print("\nSample embedding (first sentence of first batch):")
    print(batch_results[0][0][:10])  # 只打印前10维

if __name__ == "__main__":
    main()
