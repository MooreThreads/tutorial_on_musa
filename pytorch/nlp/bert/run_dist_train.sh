# run bert in 8 cards with ddp, enable amp
start_time=$(date +%s)

cur_path=`pwd`
mkdir -p "$cur_path/log"

export AMP_ON=1
export DDP_ON=1
export DEVICE='musa'
export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python -m torch.distributed.run --nproc_per_node 8 examples/sequence_labeling/task_sequence_labeling_ner_crf.py \
    > $cur_path/log/dist_train.log 2>&1 &

wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "end to end time: $e2e_time" >> ttime.log
