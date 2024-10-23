0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```bash
# download stable-diffusion-xl from huggingface
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

2. Prepare dataset
```bash

```

3. Prepare scripts
```bash
# Ubuntu system
apt update
apt install -y libgl1-mesa-glx

pip install einops toml imagesize opencv-python voluptuous transformers==4.36.1 diffusers
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
    --model_path ./stable-diffusion-xl-base-1.0 \
    --batch_size 2
```
