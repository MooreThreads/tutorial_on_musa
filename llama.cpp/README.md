## 一. Build with docker
### 推荐，可以使用llama.cpp全部功能
### 1. Start docker
启动命令可参考: [README.md](../pytorch/README.md)

### 2. Prepare scripts
```
# 
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_MUSA=ON
cmake --build build --config Release
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. hf to gguf
```
# /home/dist/models/llama-2-13b-chat-hf-fp16/ 为模型路径，修改为实际路径，完成后会在这个路径下生成一个gguf文件
python convert_hf_to_gguf.py  /home/dist/models/llama-2-13b-chat-hf-fp16/
```

### 5.cli
```
cd build/bin
# ./llama-cli -h 查看参数说明
./llama-cli -m /home/dist/models/llama-2-13b-chat-hf-fp16/llama-2-13B-chat-hf-F16.gguf -ngl 999 -n 512 -co -cnv -p "You are a helpful assistant."
# 执行该条命令加载完模型后会出一个对话框，在对话框中输入问题
```

### 6.benchmark
```
# ./llama-bench -h 查看参数说明
./llama-bench -m /home/dist/models/llama-2-13b-chat-hf-fp16/llama-2-13B-chat-hf-F16.gguf
```

## 二. pull docker
### 不推荐，只有cli和server功能,只支持amd64
### 1. pull docker and run
```
'''
ghcr.io/ggerganov/llama.cpp:full-musa: This image includes both the main executable file and the tools to convert LLaMA models into ggml and convert into 4-bit quantization. (platforms: , linux/amd64)
ghcr.io/ggerganov/llama.cpp:light-musa: This image only includes the main executable file. (platforms: , linux/amd64)
ghcr.io/ggerganov/llama.cpp:server-musa: This image only includes the server executable file. (platforms: , linux/amd64)
'''
docker pull ghcr.io/ggerganov/llama.cpp:light-musa

docker run -it -v $HOME/models:/models ghcr.io/ggerganov/llama.cpp:light-musa \
    -m /models/llama3.2_1b_q8_0.gguf -ngl 999 -n 512 -co -cnv \
    -p "You are a helpful assistant."
```
