0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download resnet50
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth

#wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth
#wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth
#wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb8_cub_20220307-57840e60.pth
```

2. Prepare dataset
```
# You need to register an account at OpenXlab official website and login by CLI.
# install OpenXlab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login  # [Access Key ID]: wqwrnxmr1qgovj5yppvm [Secret Access Key]: 4q8a1kj7grvjmpeggvelmvoxnw3owdz2ymbxlnxb
# download and preprocess by MIM, better to execute in $MMPreTrain directory.

# download ImageNet-1k
openxlab dataset get --dataset-repo OpenDataLab/ImageNet-1K && cd OpenDataLab___ImageNet-1K/raw/ && cat ImageNet-1K.tar.gz.000* > ImageNet-1K.tar.gz && tar -xvf ImageNet-1K.tar.gz
cp valprep.sh OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val
bash valprep.sh

# download cifar-10
openxlab dataset get --dataset-repo OpenDataLab/CIFAR-10 && cd OpenDataLab___CIFAR-10/raw/ && tar -zxvf cifar-10-python.tar.gz

# download cifar-100
openxlab dataset get --dataset-repo OpenDataLab/CIFAR-100 && cd OpenDataLab___CIFAR-100/raw/ && tar -zxvf cifar-100-python.tar.gz
```

3. Prepare scripts
```
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

cd package
# 目前只是编译了py310 使用驱动版本为rc3.1.0
pip install mmcv-2.1.0-cp310-cp310-linux_x86_64.whl

git clone -b 9124ebf7a285a https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -r requirements.txt
python setup.py install

git clone -b 17a886cb582 https://github.com/open-mmlab/mmpretrain.git 
cd mmpretrain
pip install -e .

vim configs/_base_/datasets/imagenet_bs32.py # 修改data_root 的值为数据所在地
vim configs/_base_/default_runtime.py        # 修改load_from 的值，放在''中

```

4. Train
```shell
# single GPU
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --auto-scale-lr --amp
                                     resnet50_8xb16_cifar100.py
                                     resnet50_8xb16_cifar10.py


# Multi-GPU DDP
export MUSA_KERNEL_TIMEOUT=3600000
bash tools/dist_train.sh configs/resnet/resnet50_8xb32_in1k.py 8 --auto-scale-lr --amp

```

5. Inference
```shell
python tools/test.py configs/resnet/resnet50_8xb32_in1k.py resnet50_8xb32_in1k_20210831-ea4938fc.pth
                                    resnet50_8xb16_cifar100.py
                                    resnet50_8xb16_cifar10.py

```
