#!/bin/bash  

if [[ $1 == "--help" ]]; then  
    echo "Usage: run.sh [TP_SIZE] [PP_SIZE] [MODEL_PATH] [HOSTFILE] [VLLM_PP_LAYER_PARTITION]"  
    echo ""  
    echo "Parameters:"  
    echo "  TP_SIZE                      Number of Tensor Parallelism"  
    echo "  PP_SIZE                      Number of Pipeline Parallelism"  
    echo "  MODEL_PATH                   Path to the model"  
    echo "  HOSTFILE                     Host file for distributed training"  
    echo "  VLLM_PP_LAYER_PARTITION      Optional partition scheme (comma-separated values); omit to skip"  
    echo ""  
    echo "Example:"  
    echo "  ./run.sh 2 4 /path/to/model /path/to/hostfile 13,12,12,12,12"  
    exit 0  
fi  

set -u  
TP_SIZE=$1  
PP_SIZE=$2  
MODEL_PATH=$3  
HOSTFILE=$4  
VLLM_PP_LAYER_PARTITION="${5:-}"
set +u  

MODEL_NAME=deepseek
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
WORLD_SIZE=$(($PP_SIZE * $TP_SIZE))
SSH_PORT=62262
RAY_PORT=63794


# For S5000 RoCE
#   MUSA_BLOCK_SCHEDULE_MODE=1
#   MCCL_IB_GID_INDEX=3

env_array=(
    MCCL_PROTOS=2
    MUSA_PRINT_ENV=1
    MUSA_HOME="/usr/local/musa"
    MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    TRITON_CACHE_DIR="/tmp/triton"
    LIBRARY_PATH="/opt/intel/oneapi/mkl/lib/intel64:${LIBRARY_PATH}"
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib"
    VLLM_NCCL_SO_PATH="/usr/local/musa/lib/libmccl.so.2"
)

if [[ -n "$VLLM_PP_LAYER_PARTITION" ]]; then  
    env_array+=(VLLM_PP_LAYER_PARTITION="$VLLM_PP_LAYER_PARTITION")  
fi  

for item in "${env_array[@]}"; do
    echo "export $item"  
    eval "export $item"  
done  

pkill -f /opt/conda/envs/py310/bin/python3
ray stop
rm -rf ${TRITON_CACHE_DIR}/*

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

set -u
  WORK_HOME="$PWD"
  EXPNAME="${MODEL_NAME}_pp${PP_SIZE}_tp${TP_SIZE}_gpus${WORLD_SIZE}"
  LOG_FILE=$WORK_HOME/output/$CURRENT_TIME/$EXPNAME.log
set +u

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

first_host=true
first_host_ip=127.0.0.1
for host in ${hostlist[@]}; do
  echo ray start $host
  ((COUNT++))
  if $first_host; then
    first_host=false
    first_host_ip=$host
    ssh -p $SSH_PORT $host "${env_array[@]} ray start --head --port=${RAY_PORT} --dashboard-host='0.0.0.0' --num-gpus 8"
    sleep 3s
  else
    ssh -p $SSH_PORT $host "${env_array[@]} ray start --address ${first_host_ip}:${RAY_PORT} --num-gpus 8" 
  fi
done

ray status

vllm serve $MODEL_PATH  \
    --trust-remote-code \
    --max-num-seqs 64 \
    --max_model_len $MAX_MODEL_LEN \
    --num-gpu-blocks-override $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION  \
    --served-model-name $MODEL_NAME \
    --distributed-executor-backend ray \
    --port 8000 \
    -tp $TP_SIZE \
    -pp $PP_SIZE 2>&1 | tee -a $LOG_FILE
