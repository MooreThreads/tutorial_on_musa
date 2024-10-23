# run bert in 8 cards with ddp, enable amp
start_time=$(date +%s)

cur_path=`pwd`
export AMP_ON=1
export DDP_ON=1
export DEVICE='musa'
export MUSA_VISIBLE_DEVICES="0"

python -m torch.distributed.run --nproc_per_node 1 examples/sequence_labeling/task_sequence_labeling_ner_crf.py \
    > $cur_path/log/train.log 2>&1 &

wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "end to end time: $e2e_time" >> ttime.log