0. Start docker  
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```shell
cd tutorial_on_musa/pytorch/nlp/bert
mkdir -p data
cd data
apt-get update && apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese
```

2. Prepare dataset
```shell
cd tutorial_on_musa/pytorch/nlp/bert/data
wget http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
tar -zxvf china-people-daily-ner-corpus.tar.gz
# 整理为：./china-people-daily-ner-corpus/{example.train,example.dev,example.test}
```

3. Prepare bert4torch
```shell
cd tutorial_on_musa/pytorch/nlp/bert
git clone https://github.com/Tongjilibo/bert4torch.git
cd bert4torch
git reset --hard ebd53d61c28295
# 每台训练节点/容器都要执行
python setup.py install
pip install torch4keras==0.2.7
cd ..
# 仅单机训练需要；多机用本目录 task_sequence_labeling_ner_crf_2node.py，无需此 cp
cp task_sequence_labeling_ner_crf.py bert4torch/examples/sequence_labeling/
```

4. Train
```shell
# 单机：在 bert4torch 下跑（需要 ./data → 本目录 data）
cd tutorial_on_musa/pytorch/nlp/bert/bert4torch
ln -sfn ../data data
bash ../run_train.sh          # 单卡
bash ../run_dist_train.sh     # 单机 8 卡

# 多机 2×8：本目录须在两机共享盘（含 data/ 与脚本）；两机免密 SSH（默认端口 62216，可用 SSH_PORT 覆盖）
cd tutorial_on_musa/pytorch/nlp/bert
bash run_dist_train_2node.sh <NODE0_IP> <NODE1_IP> 50
```

5. Inference
```shell
cd tutorial_on_musa/pytorch/nlp/bert
cp test_bert.py bert4torch/test/models/
cd bert4torch
# 先确认 test_bert.py 里 model_path 指向正确权重
python test/models/test_bert.py
```
