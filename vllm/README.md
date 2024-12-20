# Image list

- kuae1.3.0 : `registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.0-kuae1.3.0`
- kuae1.2.0 : `registry.mthreads.com/mcctest/musa-pytorch-transformer-vllm:v0.1.4-kuae1.2`

> "kuae1.x.0" is the driver version, you can check your driver version with the command `clinfo | grep 'Driver Version'`
>
> We have only verified the above images on the S4000 machine

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

# Inference Demo

```bash
python generate_chat.py -ckpt <model_path>
```

more demos(base on kuae1.2 driver version): [models](./models)

# Benchmarks

[Benchmarks](./benchmarks)

# More

[MT-Transformer-vLLM User Guide](https://docs.mthreads.com/mtt/mtt-doc-online/)
