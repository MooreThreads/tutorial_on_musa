0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download swin transformer model
# swin-v1
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth

# swin-v2
wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth

#wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth
#wget https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth
```

2. Prepare dataset
```
# download Imagenet-1k
# You need to register an account at OpenXlab official website and login by CLI.
# install OpenXlab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login  # [Access Key ID]: wqwrnxmr1qgovj5yppvm [Secret Access Key]: 4q8a1kj7grvjmpeggvelmvoxnw3owdz2ymbxlnxb
# download and preprocess by MIM, better to execute in $MMPreTrain directory.

# download ImageNet-1k
openxlab dataset get --dataset-repo OpenDataLab/ImageNet-1K && cd OpenDataLab_ImageNet-1K/raw/ && cat ImageNet-1K.tar.gz.000* > ImageNet-1K.tar.gz && tar -xvf ImageNet-1K.tar.gz
cp valprep.sh OpenDataLab_ImageNet-1K/raw/ImageNet-1K/val
bash valprep.sh

```

3. Prepare scripts
```
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
git reset --hard f82860bfb5
pip install -r requirements.txt

```

4. Convert scripts to musa
```shell
cp -r setup.py Swin-Transformer/kernels/window_process
cd Swin-Transformer/kernels/window_process

# 编译安装
FORCE_MUSA=1 python setup.py install

musa-converter -r ./Swin-Transformer -l ./Swin-Transformer/main.py

vim /opt/conda/envs/py310/lib/python3.10/site-packages/timm/data/mixup.py +23
# 与swin transformer匹配的timm包中该函数device='cuda'是默认的，在swin代码中不能自己改变device，所以需要修改该代码。
# device = target.device
```

5. Train
```shell
# single GPU
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128

# Multi-GPU DDP
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 

```

6. Inference
```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path <imagenet-path>

```
7. ThoughOut
```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --disable_amp

```