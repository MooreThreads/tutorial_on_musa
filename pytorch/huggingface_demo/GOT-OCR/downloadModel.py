from modelscope import snapshot_download
#cache_dir为指定自己本地下载目录,此处./ 为READ.md，downloadModel.py的同级目录
cache_dir = './'
#下载模型,stepfun-ai/GOT-OCR2_0是摩搭上模型id号
model_dir = snapshot_download('stepfun-ai/GOT-OCR2_0', cache_dir=cache_dir)