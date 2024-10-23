0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese

```

2. Prepare dataset
```
wget http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz && tar -zxvf china-people-daily-ner-corpus.tar.gz 

```

3. Prepare scripts
```
git clone https://github.com/Tongjilibo/bert4torch.git
cd bert4torch
git reset --hard ebd53d61c28295
python setup.py install

cp -r task_sequence_labeling_ner_crf.py bert4torch/examples/sequence_labeling
```

4. Train
```shell
# single GPU
bash run_train.sh
# Multi-GPU DDP
bash run_dist_train.sh
```

5. Inference
```shell
cp -r test_bert.py bert4torch/test/models/
python bert4torch/test/models/test_bert.py
```
