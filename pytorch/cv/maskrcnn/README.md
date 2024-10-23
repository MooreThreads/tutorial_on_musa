**Our implementation utilizes the Mask R-CNN architecture developed by Facebook AI Research.Please see [mask_rcnn](https://github.com/facebookresearch/detectron2)**

0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download R_50-_FPN model
wget https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth

# wget https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_101_FPN_1x.pth

```

2. Prepare dataset
```
# download COCO2017
mkdir -p coco/images
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip -d coco && rm annotations_trainval2017.zip
curl -L http://images.cocodataset.org/zips/train2017.zip -o train2017.zip && unzip -q train2017.zip -d coco/images && rm train2017.zip
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip && unzip -q val2017.zip -d coco/images && rm val2017.zip
curl -L http://images.cocodataset.org/zips/test2017.zip -o test2017.zip && unzip -q test2017.zip -d coco/images && rm test2017.zip

```

3. Prepare scripts
```
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip install -r requirements.txt

mkdir -p datasets/coco 
ln -s path/to/coco/train2017 datasets/coco/train2017
ln -s path/to/coco/val2017 datasets/coco/val2017
ln -s path/to/coco/test2017 datasets/coco/test2017
ln -s path/to/coco/annotations datasets/coco/annotations

```

4. build
```shell

# 编译安装
python setup.py clean build develop

```

5. Train
```shell
# single GPU
bash run_single.sh

# Multi-GPU DDP
bash run_4cards.sh
bash run_8cards.sh

```

6. Inference
```shell
bash run_test.sh

```
