import torch
import time
from transformers import pipeline

# 指定本地模型路径
model_dir = "./LLM-Research/Llama-3___2-3B-Instruct"

# 设置推理管道
pipe = pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    device_map="musa",
)

# 输入消息
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# 清除显存缓存并开始计时
torch.musa.empty_cache()
start_time = time.time()

# 推理生成
outputs = pipe(messages, max_new_tokens=256)

# 结束计时
inference_time = time.time() - start_time

# 获取生成的文本
generated_text = str(outputs[0]["generated_text"])

# 计算生成的总 token 数
total_tokens = len(pipe.tokenizer.encode(generated_text))

# 获取最大显存使用量
max_memory_allocated = torch.musa.max_memory_allocated() / (1024 ** 2)  # MB

# 输出生成的文本及性能数据
print(f"生成的文本: {generated_text}")
print(f"推理时间: {inference_time:.2f} 秒")
print(f"总生成 token 数: {total_tokens}")
print(f"最大显存使用量: {max_memory_allocated:.2f} MB")
