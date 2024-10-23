from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import time

device = "musa"  # The device to load the model onto
model_path = "./AI-ModelScope/Mistral-7B-v0___1"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare input by concatenating the conversation history manually
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Combine messages into a single input string
conversation = ""
for message in messages:
    role = message["role"]
    content = message["content"]
    conversation += f"{role.capitalize()}: {content}\n"

# Tokenize the concatenated conversation string
model_inputs = tokenizer(conversation, return_tensors="pt").to(device)
model.to(device)

# Measure inference time and memory usage
start_time = time.time()
with torch.musa.amp.autocast(enabled=(device == "musa")):  # Automatically uses MUSA AMP if available
    generated_ids = model.generate(model_inputs['input_ids'], max_new_tokens=1000, do_sample=True)
end_time = time.time()

# Decoding generated output
decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Response:", decoded)

# Calculate and print metrics
generation_time = end_time - start_time
total_tokens = generated_ids.shape[-1]
tokens_per_second = total_tokens / generation_time
print(f"Inference Time: {generation_time:.2f} seconds")
print(f"Total Generated Tokens: {total_tokens}")
print(f"Tokens per Second: {tokens_per_second:.2f}")

# Print peak memory usage (ensure MUSA memory is available and imported)
if device == "musa":
    import torch_musa  # Only necessary if torch_musa is installed
    peak_memory = torch.musa.memory_allocated()
    print(f"Peak MUSA Memory Usage: {peak_memory / 1024 ** 2:.2f} MB")
else:
    print(f"run in cpu")
