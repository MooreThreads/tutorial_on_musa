假设工作空间为本目录：`tutorial_on_musa/pytorch/nlp/bert`。

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
python setup.py install
cd ..
cp task_sequence_labeling_ner_crf.py bert4torch/examples/sequence_labeling/
```

4. Train
```shell
# 单机：在 bert4torch 下跑（需要 ./data → 本目录 data）
cd tutorial_on_musa/pytorch/nlp/bert/bert4torch
ln -sfn ../data data
bash ../run_train.sh          # 单卡
bash ../run_dist_train.sh     # 单机 8 卡

# 多机 2×8：数据在共享盘上同样按 1–3 准备，从本目录启动
cd tutorial_on_musa/pytorch/nlp/bert
bash run_dist_train_2node.sh <NODE0_IP> <NODE1_IP> 50
# SSH 端口默认 62216，可用 SSH_PORT 覆盖
```

5. Inference
```shell
cd tutorial_on_musa/pytorch/nlp/bert
cp test_bert.py bert4torch/test/models/
cd bert4torch
# 先确认 test_bert.py 里 model_path 指向正确权重
python test/models/test_bert.py
```
