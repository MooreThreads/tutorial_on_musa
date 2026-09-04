#!/bin/bash
# Multi-node bert+CRF NER DDP (2 nodes x 8 GPUs = 16 ranks)
# Uses REAL data/model from README: bert-base-chinese + china-people-daily-ner-corpus
#
# Usage (from a node that can SSH to both peers on port 62216):
#   bash run_dist_train_2node.sh <NODE0_IP> <NODE1_IP> [MAX_STEPS]
#
# Prerequisites (shared PVC, see README "多机（2×8）"):
#   WORK_DIR/data/bert-base-chinese/
#   WORK_DIR/data/china-people-daily-ner-corpus/
#   bert4torch installed; task_sequence_labeling_ner_crf_2node.py in WORK_DIR
#
# Env overrides:
#   SSH_PORT=62216 WORK_DIR=... MASTER_PORT=29501 NPROC_PER_NODE=8

set -euo pipefail

NODE0="${1:?usage: $0 <node0_ip> <node1_ip> [max_steps]}"
NODE1="${2:?usage: $0 <node0_ip> <node1_ip> [max_steps]}"
MAX_STEPS="${3:-50}"

SSH_PORT="${SSH_PORT:-62216}"
MASTER_PORT="${MASTER_PORT:-29511}"
NNODES=2
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}}"
TRAIN_PY="${WORK_DIR}/task_sequence_labeling_ner_crf_2node.py"
LOG_DIR="${WORK_DIR}/log_2node_$(date +%Y%m%d_%H%M%S)"
LOCAL_CACHE="${LOCAL_CACHE:-/tmp/bert_ddp_2node_cache}"
BATCH_SIZE="${BATCH_SIZE:-16}"

ssh_cmd() {
  local ip="$1"; shift
  ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 "$ip" "$@"
}

echo "[info] NODE0(master)=${NODE0} NODE1=${NODE1}"
echo "[info] WORK_DIR=${WORK_DIR} MAX_STEPS=${MAX_STEPS} LOG_DIR=${LOG_DIR}"
echo "[info] MASTER_PORT=${MASTER_PORT} BATCH_SIZE=${BATCH_SIZE} LOCAL_CACHE=${LOCAL_CACHE}"

# Validate shared assets (PVC)
for req in \
  "${TRAIN_PY}" \
  "${WORK_DIR}/data/bert-base-chinese/pytorch_model.bin" \
  "${WORK_DIR}/data/bert-base-chinese/config.json" \
  "${WORK_DIR}/data/bert-base-chinese/vocab.txt" \
  "${WORK_DIR}/data/china-people-daily-ner-corpus/example.train"
do
  if [ ! -e "${req}" ]; then
    echo "[fail] missing: ${req}"
    echo "       Follow README multi-node prep (model + dataset + this script)."
    exit 1
  fi
done

for ip in "${NODE0}" "${NODE1}"; do
  ssh_cmd "$ip" "mkdir -p '${LOG_DIR}' '${LOCAL_CACHE}' && test -f '${TRAIN_PY}'"
done

# Prefetch bert weights to per-node local disk (one copy per node) before torchrun
for ip in "${NODE0}" "${NODE1}"; do
  echo "[info] prefetch bert weights on ${ip} -> ${LOCAL_CACHE}"
  ssh_cmd "$ip" "bash -lc '
    set -euo pipefail
    mkdir -p \"${LOCAL_CACHE}\"
    if [ ! -f \"${LOCAL_CACHE}/bert-base-chinese/pytorch_model.bin\" ]; then
      rm -rf \"${LOCAL_CACHE}/bert-base-chinese\"
      cp -a \"${WORK_DIR}/data/bert-base-chinese\" \"${LOCAL_CACHE}/bert-base-chinese\"
    fi
    ls -lh \"${LOCAL_CACHE}/bert-base-chinese/pytorch_model.bin\"
  '"
done

# best-effort cleanup (bracket pattern avoids killing this shell)
for ip in "${NODE0}" "${NODE1}"; do
  ssh_cmd "$ip" "pkill -f '[t]ask_sequence_labeling_ner_crf_2node' || true; pkill -f 'torchrun.*[t]ask_sequence_labeling_ner_crf_2node' || true" || true
done
sleep 2

launch_one() {
  local ip="$1"
  local node_rank="$2"
  local logfile="${LOG_DIR}/rank${node_rank}.${ip}.log"
  ssh_cmd "$ip" "bash -lc '
    set -euo pipefail
    cd \"${WORK_DIR}\"
    export MASTER_ADDR=${NODE0}
    export MASTER_PORT=${MASTER_PORT}
    export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=\"${WORK_DIR}:${WORK_DIR}/bert4torch:${WORK_DIR}/pydeps:\${PYTHONPATH:-}\"
    nohup torchrun \\
      --nnodes=${NNODES} \\
      --nproc_per_node=${NPROC_PER_NODE} \\
      --node_rank=${node_rank} \\
      --master_addr=${NODE0} \\
      --master_port=${MASTER_PORT} \\
      \"${TRAIN_PY}\" --max-steps ${MAX_STEPS} --log-interval 5 --batch-size ${BATCH_SIZE} \\
      --data-dir ./data --local-cache ${LOCAL_CACHE} \\
      > \"${logfile}\" 2>&1 &
    echo STARTED_PID=\$! LOG=${logfile}
    sleep 1
    head -5 \"${logfile}\" || true
  '"
}

launch_one "${NODE0}" 0
launch_one "${NODE1}" 1

echo "[info] waiting for completion (poll rank0 log)..."
LOG0="${LOG_DIR}/rank0.${NODE0}.log"
# bert+CRF load can take minutes; allow up to ~20 min
for i in $(seq 1 240); do
  if ssh_cmd "${NODE0}" "grep -q '\\[bert-crf-2node\\] DONE' '${LOG0}' 2>/dev/null"; then
    echo "[ok] finished"
    ssh_cmd "${NODE0}" "tail -40 '${LOG0}'"
    echo "LOG_DIR=${LOG_DIR}"
    exit 0
  fi
  if ssh_cmd "${NODE0}" "grep -Eq 'ChildFailedError|Traceback \\(most recent call last\\)|FileNotFoundError' '${LOG0}' 2>/dev/null"; then
    echo "[fail] error detected in rank0 log"
    ssh_cmd "${NODE0}" "tail -100 '${LOG0}'" || true
    exit 1
  fi
  sleep 5
done

echo "[fail] timeout waiting for DONE"
ssh_cmd "${NODE0}" "tail -100 '${LOG0}'" || true
exit 1
