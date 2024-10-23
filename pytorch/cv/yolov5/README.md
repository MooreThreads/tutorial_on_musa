0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download yolov5m
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt

#wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
```

2. Prepare dataset
```
# download COCO2017
mkdir -p coco/images
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d ./ && rm coco2017labels.zip
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

git clone https://github.com/ultralytics/yolov5.git -b v6.0
cd yolov5
pip install -r requirements.txt
# yolo执行期间需要Arial.ttf文件，为防止因网络问题导致执行失败，提前将Arial.ttf文件放到执行位置
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf && mkdir -p /root/.config/Ultralytics && mv Arial.ttf /root/.config/Ultralytics/Arial.ttf
```

4. Convert scripts to musa
```
cd ..
musa-converter -r ./yolov5 -l ./yolov5/train.py
```

5. Train
```shell
cd yolov5
# single GPU
python train.py --data data/coco.yaml --cfg models/yolov5s.yaml --weights ./yolov5s.pt --batch-size 64 --epochs 300 --device 0
                                                   yolov5m                  yolov5m                 40
                                                   yolov5l                  yolov5l                 24
                                                   yolov5x                  yolov5x                 16

# Multi-GPU DDP
export MUSA_KERNEL_TIMEOUT=3600000
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 25555 train.py --batch 384 --cfg models/yolov5m.yaml --data data/coco.yaml --epochs 300 --weights  ./yolov5m.pt --hyp data/hyps/hyp.scratch-high.yaml --device 0,1,2,3,4,5,6,7

```

6. Inference
```shell
python val.py --data data/coco.yaml --weight runs/train/exp4/weights/best.pt --device 0

python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/LNwODJXcvt4'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

```

7. FAQ

numpy没有int类型
具体报错：
module 'numpy' has no attribute 'int'
解决方法：numpy在1.24及以后版本移除了numpy.int 因此需要选择1.24以下版本

Pillow版本问题：
具体报错:
'FreeTypeFont' object has no attribute 'getsize'
解决方法：使用pillow 9.5版本