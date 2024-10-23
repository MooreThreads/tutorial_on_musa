0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
apt install git-lfs
git lfs install
git clone https://huggingface.co/openai/whisper-small

# git clone https://huggingface.co/openai/whisper-large
# git clone https://huggingface.co/openai/whisper-tiny

```
2. Download dataset
```shell
python download.py
```

2. Prerequisites
```shell
pip install -r requirements.txt
```

4. Train
```shell
# Single GPU
bash train_whisper.sh

# Multi-GPU DDP
bash dist_train_whisper.sh
```

4. Test
```shell
python test_whisper.py --models_dir <models-dir> --data_dir <datasets-dir>
```