from modelscope import snapshot_download
#cache_dir为指定自己本地下载目录,此处./ 为READ.md，downloadModel.py的同级目录
cache_dir = './'
model_dir = snapshot_download('swift/llava-interleave-qwen-0.5b-hf', cache_dir=cache_dir)