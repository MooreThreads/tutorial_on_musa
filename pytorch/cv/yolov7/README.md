0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
# download yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

#wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
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

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
git reset --hard a207844b1c
pip install -r requirements.txt
vim data/coco.yaml  #修改7,8,9的./coco 为/data/coco

# yolo执行期间需要Arial.ttf文件，为防止因网络问题导致执行失败，提前将Arial.ttf文件放到执行位置
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf && mkdir -p /root/.config/Ultralytics && mv Arial.ttf /root/.config/Ultralytics/Arial.ttf
```

4. Convert scripts to musa
```
musa‑converter ‑r ./yolov7 ‑l ./yolov7/train.py
```

5. Train
```shell
# single GPU
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
                                                    16                                                        yolov7x.yaml                    yolov7x 

# Multi-GPU DDP
export MUSA_KERNEL_TIMEOUT=3600000
export LOCAL_RANK=0

python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 64 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights '' --name yolov7x --hyp data/hyp.scratch.p5.yaml
```

6. Inference
```shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val

python detect.py --weights yolov7.pt --source 0                               # webcam
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
