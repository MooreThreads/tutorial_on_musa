import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import time

model_path = "./swift/llava-interleave-qwen-0___5b-hf"  

# 记录显存使用情况
def get_gpu_memory_info():
    torch.musa.synchronize()
    return {
        "max_memory_allocated": torch.musa.max_memory_allocated() / 1024 ** 2,  # MB
        "memory_reserved": torch.musa.memory_reserved() / 1024 ** 2,  # MB
        "memory_allocated": torch.musa.memory_allocated() / 1024 ** 2,  # MB
    }

# 获取 GPU 的总显存
def get_total_gpu_memory():
    device_properties = torch.musa.get_device_properties(0)
    total_memory = device_properties.total_memory / 1024 ** 2  # MB
    return total_memory

# 加载模型和处理器
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_path)

# 定义对话历史并使用 `apply_chat_template` 获取正确格式的提示
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 使用本地或在线图片
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)

# 记录输入图片大小
image_size = raw_image.size  # (width, height)

# 处理输入
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# 记录推理开始时间
start_time = time.time()

# 推理过程
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# 记录推理结束时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time

# 计算生成的 token 数（从第2个位置开始，因为第1个通常是特殊 token）
total_tokens = output.size(1) - 2

# 计算每秒生成的 token 数
tokens_per_second = total_tokens / inference_time

# 获取显存使用信息
memory_info = get_gpu_memory_info()
max_memory_used = memory_info['max_memory_allocated']

# 获取 GPU 总显存
total_memory = get_total_gpu_memory()

# 计算显存利用率
memory_utilization = (max_memory_used / total_memory) * 100

# 打印性能指标
print(f"推理时间: {inference_time:.2f} 秒")
print(f"总生成 token 数: {total_tokens}")
print(f"每秒生成 token 数: {tokens_per_second:.2f} tokens/s")
print(f"输入图片大小: {image_size[0]}x{image_size[1]}")
print(f"最大显存使用量: {max_memory_used:.2f} MB")
print(f"最大显存利用率: {memory_utilization:.2f}%")

# 输出推理结果
print(processor.decode(output[0][2:], skip_special_tokens=True))
