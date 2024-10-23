#!/bin/bash


if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <original_model_path> <tp_size> <model_type(optional)>"
    exit 1
fi

original_model_path=$1
tp_size=$2

# 目前支持llama, mistral, chatglm2, baichuan, qwen, qwen2, yayi；
# mistral和llama同构，model_type也可以填llama；
# chatglm2和chatglm3同构，model_type都是chatglm2；
# Qwen1.5和Qwen2同构，model_type都是qwen2，但是Qwen的依然是qwen；
# baichuan, baichuan2，model_type都是baichuan
model_type=$3

model_name=$(basename "$original_model_path")
model_dir=$(dirname "$original_model_path")
saved_dir="${model_dir}/${model_name%.*}-tp${tp_size}-convert"
mkdir -p "$(dirname "$saved_dir")"

if [ -z "$model_type" ]; then
    python -m mttransformer.convert_weight --in_file "$original_model_path" --saved_dir "$saved_dir" -tp $tp_size
else
    python -m mttransformer.convert_weight --in_file "$original_model_path" --saved_dir "$saved_dir" -tp $tp_size --model_type $model_type
fi


if [ $? -eq 0 ]; then
    echo "Conversion successful. Weights saved to $saved_dir"
else
    echo "Conversion failed."
    exit 1
fi
