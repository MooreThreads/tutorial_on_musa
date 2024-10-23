0 Start docker 启动命令可参考:  [README.md](../README.md)

1 Prepare model 
```
 pip install modelscope
 python downloadModel.py
```
2 Prepare depandents(Ubuntu_environment)
```
pip install opencv-python
apt-get update
apt-get install -y libgl1-mesa-glx
pip install matplotlib
conda install -c conda-forge libstdcxx-ng
pip install tiktoken
pip install verovio
pip install 'accelerate>=0.26.0'
```

3 Convert scripts to musa

```
musa‑converter ‑r ./stepfun-ai/GOT-OCR2_0 ‑l ./stepfun-ai/GOT-OCR2_0/modeling_GOT.py
```

4 Download infer_test video
- 访问[Video-text-to-text (huggingface.co)](https://huggingface.co/docs/transformers/tasks/video_text_to_text) 下载测试视频cats_1.mp4
- 把下载的cats_1.mp4视频文件放入./stepfun-ai/GOT-OCR2_0文件下

5 Run run_model.py
```
cp run_model.py ./stepfun-ai/GOT-OCR2_0
python ./stepfun-ai/GOT-OCR2_0/run_model.py
```

