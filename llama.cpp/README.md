## 一. Build with docker
### 推荐，可以使用llama.cpp全部功能
### 1. Start docker
启动命令可参考: [README.md](../pytorch/README.md)

### 2. Prepare scripts
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_MUSA=ON
cmake --build build --config Release -- -j$(nproc)
```

### 3. Install requirements
```
# 注意需要保证python环境 ≥ 3.9
pip install -r requirements.txt
```
### 4. model download
推荐用modelscope下载模型（[Modelscope-模型库](https://www.modelscope.cn/models)），也可以用huggingface下载模型
```

# 这里给出llama-2-13b-chat-hf-fp16的下载方式
mkdir -p /home/dist/models/ && cd /home/dist/models/
git lfs install
git clone https://www.modelscope.cn/ydyajyA/Llama-2-13b-chat-hf.git
```

> 下载可能会遇到错误：
> 1. **确保 Git LFS 正确安装和配置**
>
> 首先确保 Git LFS 已经正确安装。在 Linux 上，你可以通过以下命令来检查 Git LFS 的版本，确认其是否已安装：
> ```
> git lfs version
> ```
> 如果 Git LFS 还没有安装，你需要根据你的 Linux 发行版来安装它。例如，对于基于 Debian 的系统（如 Ubuntu），> 可以使用：
> ```
> sudo apt update
> sudo apt install git-lfs
> ```
> 安装完成后，需要执行：
> ```
> git lfs install
> ```
> 这个命令会确保 Git LFS 的钩子被正确设置。
> 
> 2. **重新拉取 Git LFS 文件**
> 
> 如果 Git LFS 已经安装，你可以尝试重新拉取有问题的文件。在仓库目录内执行：
> ```
> git lfs pull
> ```
> 这会尝试重新下载所有由 Git LFS 管理的大文件。
> 
> 3. **分步获取和检出文件**
> 
> 如果直接拉取不起作用，你可以尝试使用 fetch 和 checkout 命令来分步操作：
> ```
> git lfs fetch --all
> git lfs checkout
> ```
> 这些命令会确保从远程仓库获取所有的 LFS 对象，并尝试将这些对象在本地检出。

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
