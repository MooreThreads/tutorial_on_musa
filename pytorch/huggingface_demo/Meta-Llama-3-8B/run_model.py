import transformers
import torch
import time

# 模型 ID，使用本地 Meta-Llama-3-8B 模型路径
model_id = "./LLM-Research/Meta-Llama-3-8B"

# 设置文本生成管道，使用 bfloat16 精度，并选择设备
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="musa"
)

# 输入文本
input_text = "Hey how are you doing today?"

# 开始计时
start_time = time.time()

# 生成响应
response = pipeline(input_text, max_new_tokens=50)

# 结束计时
end_time = time.time()
inference_time = end_time - start_time

# 获取生成的总 token 数
generated_text = response[0]['generated_text']
total_tokens = len(generated_text.split())

# 获取最大显存使用量
max_memory_allocated = torch.musa.max_memory_allocated() / (1024**2)  # MB

# 输出结果
print(f"Response: {generated_text}")
print(f"Inference time: {inference_time:.2f} seconds")
print(f"Total tokens generated: {total_tokens}")
print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
