#!/bin/bash
# Date: 2024-10-31
# Author: create by GPU-Genius
# Version: 1.0
# Function: Take Qwen2-0.5B-Instruct model as an example to automatically verify whether the environment in the vLLM container is normal. You can execute: bash ./vllm_check.sh .



function vllm_docker_test() {
    current_time=$(date +"%Y%m%d-%H%M%S")
    CONTAINER_NAME="vllm_mtt_test_${current_time}"
    MODEL_NAME=Qwen2-0.5B-Instruct
    TP=1

    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <driver_version>, eg: 1.2 , 1.3 (More: clinfo | grep 'Driver Version')"
        echo ""
        exit 1
    fi
    driver_version=$1

    image_name="registry.mthreads.com/mcctest/musa-pytorch-transformer-vllm:v0.1.4-kuae${driver_version}"
    # 1. Create a new container validation directly
    docker create -it --privileged --net host --name=${CONTAINER_NAME} -w /workspace --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ${image_name} /bin/bash
    docker start ${CONTAINER_NAME} 2> /dev/null
    # 2. Execute commands in the vllm container
    COMMANDS_BASH=(
        "mkdir -p /data/mtt/models/${MODEL_NAME} /data/mtt/models_convert && git clone https://www.modelscope.cn/Qwen/${MODEL_NAME}.git /data/mtt/models/${MODEL_NAME}"
        "git clone https://github.com/MooreThreads/examples_on_musa.git"
        "export LD_LIBRARY_PATH=/usr/local/musa/lib::/home/opt/openmpi/lib && /opt/conda/envs/py38/bin/python -m mttransformer.convert_weight --in_file /data/mtt/models/$MODEL_NAME --saved_dir /data/mtt/models_convert/$MODEL_NAME-fp16-tp$TP-convert -tp $TP"
        "export PYTHONPATH=/home/workspace/vllm_mtt; export LD_LIBRARY_PATH=/usr/local/musa/lib::/home/opt/openmpi/lib; /opt/conda/envs/py38/bin/python /workspace/examples_on_musa/vllm/generate_chat.py -ckpt /data/mtt/models_convert/$MODEL_NAME-fp16-tp$TP-convert"
    )

    ENV_FLAG=0
    for cmd in "${COMMANDS_BASH[@]}"; do
        docker exec -it ${CONTAINER_NAME} /bin/bash -c "$cmd" 2> /dev/null
        StATUS=$?
        if [ $StATUS -ne 0 ]; then
            echo "There are problems in container: ${CONTAINER_NAME}. Please check it manually"
            docker stop ${CONTAINER_NAME} > /dev/null 2>&1
            docker rm ${CONTAINER_NAME} > /dev/null 2>&1
            ENV_FLAG=1
            break
        fi
    done

    # 3. Validate environment with "ENV_FLAG"
    if [ $ENV_FLAG -eq 0 ]; then
        echo -e "\033[32mIt has been verified that there is no problem for vLLM environment related operations based on ${image_name} images \033[0m"
        docker stop ${CONTAINER_NAME} > /dev/null 2>&1
        docker rm ${CONTAINER_NAME} > /dev/null 2>&1
    fi
}

vllm_docker_test




