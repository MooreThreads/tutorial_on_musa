import os
import time
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# 本地模型路径
model_dir = './Qwen/Qwen-VL-Chat'

# 检查路径是否存在
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model path {model_dir} does not exist!")

torch.manual_seed(1234)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 加载模型并设定FP16精度
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="musa", trust_remote_code=True, fp16=True).eval()

# 生成配置
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# 第一轮对话 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么'},
])

# 清除显存缓存并开始计时
torch.musa.empty_cache()
start_time = time.time()

# 推理生成
response, history = model.chat(tokenizer, query=query, history=None)

# 计算推理时间
inference_time = time.time() - start_time
# 计算生成的总token数
total_tokens = len(tokenizer.encode(response))
# 获取最大显存使用量
max_memory_allocated = torch.musa.max_memory_allocated() / (1024 ** 2)  # MB

print(f"模型响应: {response}")
print(f"推理时间: {inference_time:.2f} 秒")
print(f"总生成 token 数: {total_tokens}")
print(f"最大显存使用量: {max_memory_allocated:.2f} MB")

# 第二轮对话 2nd dialogue turn
start_time = time.time()
response, history = model.chat(tokenizer, '输出击掌的检测框', history=history)
inference_time = time.time() - start_time
total_tokens = len(tokenizer.encode(response))
max_memory_allocated = torch.musa.max_memory_allocated() / (1024 ** 2)  # MB

print(f"模型响应: {response}")
print(f"推理时间: {inference_time:.2f} 秒")
print(f"总生成 token 数: {total_tokens}")
print(f"最大显存使用量: {max_memory_allocated:.2f} MB")

# 画出检测框并保存图片
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('output_chat.jpg')
