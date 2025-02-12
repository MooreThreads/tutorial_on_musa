#  Image list

|KuaE version |	 Driver version | Driver date |	Docker image |
| ---- | --- | --- | ----|
|1.3.0|	2.7.0| 20241017	|	registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.0-kuae1.3.0|
|1.2.0|	1.2.0|	 |	registry.mthreads.com/mcconline/mtt-vllm-public:v0.1.4-kuae1.2.0|

> Please refer to [README](../pytorch/README.md) for the method to check the driver.
>
> We have only verified the above images on the S4000 machine

# Start Docker Container
**请按照实际驱动环境，参考上述表格，更换启动docker命令中镜像名**
```bash
# Please select an appropriate Docker image based on the driver version and refer to the content above.
sudo docker run -it --privileged --net host --name=vllm_mtt_test -w /workspace -v /data/mtt/:/data/mtt/ --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.0-kuae1.3.0 /bin/bash
```
> The default model path is stored at /data/mtt, if you do not have access to /data, the directory map can be replaced with < customed_directory >:/data/mtt/ or whatever you prefer

# Check vLLM Image

Assume that your host environment has been configured successfully via [setup_musa](../setup_musa), including the driver, mt-container-toolkit: 

```bash
# (host env)
# Usage: bash ./check_vllm_image/vllm_check.sh <driver_version>
bash ./check_vllm_image/vllm_check.sh 1.2
```



# Convert Weight

```bash
bash ./convert_weight.sh -ckpt <model_path> -tp <tensor-parallel-size>
```

> <convert_model_dir>: generated from the same level of <model_path>
> 



# Inference Demo

```bash
python generate_chat.py -ckpt <model_path>
```

more demos(base on kuae1.2 driver version): [models](./models)



# Benchmarks

[Benchmarks](./benchmarks)



#  More

[MT-Transformer-vLLM User Guide](https://docs.mthreads.com/mtt/mtt-doc-online/)

