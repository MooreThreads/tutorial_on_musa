0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download yolov8n
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

#wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
#wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt
```

2. Prepare dataset
```
# download COCO2017
mkdir -p /data/coco/images
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d /data && rm coco2017labels.zip
curl -L http://images.cocodataset.org/zips/train2017.zip -o train2017.zip && unzip -q train2017.zip -d /data/coco/images && rm train2017.zip
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip && unzip -q val2017.zip -d /data/coco/images && rm val2017.zip
curl -L http://images.cocodataset.org/zips/test2017.zip -o test2017.zip && unzip -q test2017.zip -d /data/coco/images && rm test2017.zip

```

3. Prepare scripts
```
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

git clone https://github.com/ultralytics/ultralytics.git
musa-converter -r ultralytics/ -l ultralytics/test/test_python.py
cd ultralytics && mkdir -p runs/detect/train
# 由于 ultralytics 的检测pytorch 版本的方法有问题，导致torch_musa的amp没有正确使用，所以需要手动修改一下
vim ultralytics/utils/torch_utils.py +104
if TORCH_2_4
pip install -e .
vim ultralytics/cfg/datasets/coco.yaml #第11行 ../datasets/coco 为/data/coco

# yolo执行期间需要Arial.ttf文件，为防止因网络问题导致执行失败，提前将Arial.ttf文件放到执行位置
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf && mkdir -p /root/.config/Ultralytics && mv Arial.ttf /root/.config/Ultralytics/Arial.ttf

```

4. Train
```shell
# single GPU
vim train.py

from ultralytics import YOLO
import torch
import torch_musa

model = YOLO("./yolov8n.pt") #yolov8s.pt/yolov8m.pt/yolov8l.pt/yolov8x.pt
model.info()

# Start training on your custom dataset
# 有关更多参数信息可参考https://docs.ultralytics.com/modes/train/#train-settings
model.train(
            data="./ultralytics/cfg/datasets/coco.yaml",
            epochs=100,
            imgsz=640,
            batch=32)

# Evaluate model performance on the validation set
metrics = model.val()

# Multi-GPU DDP
export MUSA_KERNEL_TIMEOUT=3600000

vim dist_train.py

from ultralytics import YOLO
import torch
import torch_musa

model = YOLO("./yolov8n.pt") #yolov8s.pt/yolov8m.pt/yolov8l.pt/yolov8x.pt
model.info()

# Start training on your custom dataset
model.train(data="./ultralytics/cfg/datasets/coco.yaml", epochs=100, imgsz=640, batch=128, device='0,1,2,3')

# Evaluate model performance on the validation set
metrics = model.val(data="./ultralytics/cfg/datasets/coco.yaml")

# RUN 
RANK=8 python dist_train.py
```

5. Inference
```shell
# Validate
# 有关val更多参数信息参考https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation
vim val.py

from ultralytics import YOLO
import torch
import torch_musa

model = YOLO("./yolov8n.pt") #yolov8s.pt/yolov8m.pt/yolov8l.pt/yolov8x.pt
model.info()
metrics = model.val(
    data='./ultralytics/cfg/datasets/coco.yaml',  # 训练所用的数据，见下方
    imgsz=640,  # 验证时所设置的图片大小
    conf=0.25,  # 验证时置信度阈值
    iou=0.6,  # 验证时的IOU阈值
    device='0',  # 验证使用的GPU ID，通常单卡即可
)

------------------------------------------------------------------------------------------------
# Predict
vim predict.py

from ultralytics import YOLO
import torch
import torch_musa

model = YOLO("./yolov8n.pt") #yolov8s.pt/yolov8m.pt/yolov8l.pt/yolov8x.pt
model.info()

results = model("path/to/image.jpg")

```
