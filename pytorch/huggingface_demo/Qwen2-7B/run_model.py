from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import time

device = "musa"  # 使用MUSA设备加载模型
model_path = "./Qwen2-7B/Qwen/Qwen2-7B"  # 本地模型路径

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="musa"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置输入前缀
prefix = "北京是中国的首都"
model_inputs = tokenizer([prefix], return_tensors="pt").to(device)

# 开始计时
start_time = time.time()

# 生成文本
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=400,
    repetition_penalty=1.15
)

# 结束计时
end_time = time.time()
inference_time = end_time - start_time

# 计算生成的总 token 数
total_tokens = generated_ids.shape[1] - model_inputs.input_ids.shape[1]

# 获取最大显存使用量
max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB

# 解析输出
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出结果
print(response)
print(f"推理时间: {inference_time:.2f} 秒")
print(f"总生成 token 数: {total_tokens}")
print(f"最大显存使用量: {max_memory_allocated:.2f} MB")
