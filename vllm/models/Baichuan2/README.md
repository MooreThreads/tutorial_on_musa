# Baichuan2

## 1. Supported Models

-  Baichuan2-13B-Chat

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
git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-13B-Chat.git
```
> If you encounter any issues while downloading the model, please refer to the [README](../../../llama.cpp/README.md).

### 2.3 Weight Conversion

```bash
python -m mttransformer.convert_weight \
	--in_file /data/mtt/models/Baichuan2-13B-Chat/ \
	--saved_dir /data/mtt/models_convert/Baichuan2-13B-Chat-fp16-tp1-convert/ \
	--tensor-para-size 1
```
### 2.4 Prepare the chat template file
You can save this file as `template_baichuan2.jinja`.
```
{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<reserved_106>' + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<reserved_107>' + message['content'] -}}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
    {{- '<reserved_107>' -}}
{% endif %}
```

## 3. Inference

- start server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /data/mtt/models_convert/Baichuan2-13B-Chat-fp16-tp1-convert/ \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    -pp 1 \
    --block-size 64 \
    --max-model-len 2048 \
    --disable-log-stats \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --device "musa"
    --chat-template ./template_baichuan.jinja
```

- send message

```bash
curl http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/data/mtt/models_convert/Baichuan2-13B-Chat-fp16-tp1-convert/",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
}'
```
