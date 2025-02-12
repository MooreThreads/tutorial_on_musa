#!/bin/bash
# Date: 2024-10-31
# Author: create by GPU-Genius
# Version: 1.0
# Function: Take Qwen2-0.5B-Instruct model as an example to automatically verify whether the environment in the vLLM container is normal. You can execute it in container enviroment: bash ./vllm_check.sh .

MODEL_NAME=Qwen2-0.5B-Instruct
TP=1

mkdir -p /data/mtt/models/${MODEL_NAME} /data/mtt/models_convert 
git clone https://www.modelscope.cn/Qwen/${MODEL_NAME}.git /data/mtt/models/${MODEL_NAME}
python -m mttransformer.convert_weight --in_file /data/mtt/models/$MODEL_NAME --saved_dir /data/mtt/models_convert/$MODEL_NAME-fp16-tp$TP-convert -tp $TP
python ../generate_chat.py -ckpt /data/mtt/models_convert/$MODEL_NAME-fp16-tp$TP-convert




