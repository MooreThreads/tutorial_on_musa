# Llama-3

## 1. Supported Models

- Llama-3-8B

## 2. Setup

### 2.1 Start Docker Container

```bash
sudo docker run -it --privileged --net host --name=vllm_mtt_test -w /workspace -v /data/mtt/:/data/mtt/ --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g registry.mthreads.com/mcctest/musa-pytorch-transformer-vllm:v0.1.4-kuae1.2 /bin/bash
```

> The default model path is stored at `/data/mtt`, if you do not have access to `/data`, the directory map can be replaced with `< customed_directory >:/data/mtt/` or whatever you prefer

### 2.2 Clone Model

```bash
# 1. create directory
mkdir -p /data/mtt/models /data/mtt/models_convert

# 2. clone model
cd /data/mtt/models
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3-8B-Instruct.git
```

### 2.3 Weight Conversion

```bash
python -m mttransformer.convert_weight \
	--in_file /data/mtt/models/Meta-Llama-3-8B-Instruct/ \
	--saved_dir /data/mtt/models_convert/Meta-Llama-3-8B-Instruct-fp16-tp1-convert/ \
	--tensor-para-size 1
```

## 3. Inference

- start server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /data/mtt/models_convert/Meta-Llama-3-8B-Instruct-fp16-tp1-convert/ \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    -pp 1 \
    --block-size 64 \
    --max-model-len 4096 \
    --disable-log-stats \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --device "musa"
```

- send message

```bash
curl http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/data/mtt/models_convert/Meta-Llama-3-8B-Instruct-fp16-tp1-convert/",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
}'
```