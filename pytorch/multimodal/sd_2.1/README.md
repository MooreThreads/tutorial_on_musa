0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```bash
# download  stabilityai/stable-diffusion-2-1 from huggingface
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
```

2. Prepare dataset
```bash

```

3. Prepare scripts
```bash
# Ubuntu system
apt update
apt install -y libprotobuf-dev protobuf-compiler

pip install peft datasets diffusers transformers==4.36.1
```

4. Convert scripts to musa
```bash

```

5. Train
```bash

```

6. Inference
```bash
python sd_inference.py \
    --model_path ./stable-diffusion-2-1 \
    --batch_size 2
```
