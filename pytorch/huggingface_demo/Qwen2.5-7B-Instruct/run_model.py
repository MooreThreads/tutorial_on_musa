from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# 模型名称
model_name = "./Qwen/Qwen2___5-7B-Instruct"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="musa"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 提示文本
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 准备输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 开始计时
start_time = time.time()

# 生成响应
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 结束计时
end_time = time.time()
inference_time = end_time - start_time

# 获取生成的总 token 数
total_tokens = generated_ids.shape[1] - model_inputs['input_ids'].shape[1]

# 获取最大显存使用量
max_memory_allocated = torch.musa.max_memory_allocated() / (1024**2)  # MB

# 解码生成的 ID
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出结果
print(f"Response: {response}")
print(f"Inference time: {inference_time:.2f} seconds")
print(f"Total tokens generated: {total_tokens}")
print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
