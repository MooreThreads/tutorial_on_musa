0. Start docker
启动命令可参考: [README.md](../../README.md)

1. Prepare model
```
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
```

2. Prepare env
```
conda config --add channels conda-forge
conda update --all -y

```

3. Prepare scripts
```
conda config --add channels conda-forge
conda update --all -y 

git clone https://github.com/ultralytics/yolov5.git -b v6.0
cd yolov5 && pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:/path/to/yolov5

# 若运行 第5步报错找不到libmusa_kernels.so，可以添加以下环境变量 
# export LD_LIBRARY_PATH=/path/to/torch_musa/lib/:$LD_LIBRARY_PATH
# 例如 export LD_LIBRARY_PATH=/opt/conda/envs/py310/lib/python3.10/site-packages/torch_musa/lib/:$LD_LIBRARY_PATH

```

4. Build
```shell

cd cpp
python load_model.py --models path/to/yolov5m.pt
bash build.sh

```

5. Inference
```shell
./build/example-app yolov5m_jit.pt

```