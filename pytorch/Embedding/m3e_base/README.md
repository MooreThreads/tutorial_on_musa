0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prerequisites
```shell
pip install -r requirements.txt

pip install -U huggingface_hub
```
2. export env
```shell
export HF_ENDPOINT=https://hf-mirror.com
```

3. Test
```shell
python perf_m3e_base.py
```