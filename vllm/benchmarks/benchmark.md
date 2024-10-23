

##  Weight Convert

```shell
bash ../convert_weight.sh <model_path> <tensor-parallel-size>
```

> converted model path: ` /data/mtt/models_convert/<convert_model_dir>`



## Download Dataset

```shell
wget -P /home/workspace/vllm_mtt/benchmarks https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

> you can also download it in modescope: [数据集详情页 · 魔搭社区](https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split)

## Benchmark_offline

```shell
cd /home/workspace/vllm_mtt/benchmarks
python benchmark_throughput.py  \
	--dataset ShareGPT_V3_unfiltered_cleaned_split.json  \
	--model /data/mtt/models_convert/<convert_model_dir>/  \
	--trust-remote-code  \
	--output-len <output-len>  \
	--kv-cache-dtype "auto"  \
	--device "musa"  \
	--tensor-parallel-size <tensor-parallel-size>
```



##  Benchmark_serving

- start server

```shell
python -m vllm.entrypoints.openai.api_server  \
	--model /data/mtt/models_convert/<convert_model_dir>/  \
    --trust-remote-code  \
    --tensor-parallel-size <tensor-parallel-size>  \
    --block-size 64  \  
    --disable-log-stats  \
    --disable-log-requests  \
    --device "musa"
```

- send requests

```shell
cd /home/workspace/vllm_mtt/benchmarks
python benchmark_serving.py  \ 
	--backend <backend> # By default <backend> is vllm
	--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json  \
	--dataset-name sharegpt  \
    --model /data/mtt/models_convert/<convert_model_dir>  \
    --trust-remote-code
    --num-prompts <num_prompts> # By default <num_prompts> is 1000
```
