0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```bash
# download stable-diffusion-v1-5
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

2. Prepare dataset
```bash
# download konyconi style dataset from the following website
https://civitai.com/models/52697/tutorial-konyconi-style-lora-konyconi
```

3. Prepare scripts
```bash
# Ubuntu system
apt update
apt install -y libprotobuf-dev protobuf-compiler

pip install peft datasets diffusers transformers==4.36.1
pip install libcst==1.1.0
```

4. Convert scripts to musa
```bash
```

5. Train

需要设定accelerate的配置，利用命令accelerate config来进行配置，设置完毕之后会提示配置文件保存在

~/.cache/huggingface/accelerate/default_config.yaml

5.1 单卡微调

这里注意num_machines和num_processes都是1
```bash
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_MUSA
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

完成上面的设定之后，我们使用单卡微调的脚本run_lora_single_gpu.sh

```bash
bash run_lora_single_gpu.sh
```

5.2 单机8卡微调

这里注意num_machines是1，num_processes都是8.
```bash
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_MUSA
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

完成上面的设定之后，我们使用8卡微调的脚本run_lora_single_gpu.sh
```bash
bash run_lora_multi_gpu.sh
```

6. Inference
```bash
python sd_inference.py \
    --model_path ./stable-diffusion-v1-5 \
    --batch_size 2
```
