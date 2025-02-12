# Qwen2

## 1. Supported Models

- Qwen2-7B
- Qwen2-72B

## 2. Setup

### 2.1 Start Docker Container

Refer to the [README.md](../../README.md)

### 2.2 Clone Model

```bash
# 1. create directory
mkdir -p /data/mtt/models /data/mtt/models_convert

# 2. clone model
cd /data/mtt/models
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2-7B-Instruct.git
# git clone https://www.modelscope.cn/Qwen/Qwen2-72B-Instruct.git
```
> If you encounter any issues while downloading the model, please refer to the [README](../../../llama.cpp/README.md).
### 2.3 Weight Conversion

```bash
python -m mttransformer.convert_weight \
	--in_file /data/mtt/models/Qwen2-7B-Instruct/ \
	--saved_dir /data/mtt/models_convert/Qwen2-7B-Instruct-fp16-tp1-convert/ \
	--tensor-para-size 1
# If the --in_file is set to Qwen2-72B-Instruct, it is recommended to set the --tensor-parallel-size parameter to 8 
```

## 3. Inference

- start server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /data/mtt/models_convert/Qwen2-7B-Instruct-fp16-tp1-convert/ \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    -pp 1 \
    --block-size 64 \
    --max-model-len 4096 \
    --disable-log-stats \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --device "musa"
# If the --model is passed Qwen2-72B-Instruct, then the --tensor-parallel-size is specified as 4 or 8, which must match the value of --tensor-parallel-size when the model weights are converted
```

- send message

```bash
curl http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/data/mtt/models_convert/Qwen2-7B-Instruct-fp16-tp1-convert/",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
}'
```
