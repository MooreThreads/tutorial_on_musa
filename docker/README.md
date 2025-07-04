# Docker 容器调试与运行指南

## 1. Docker 容器状态分析 - Exited (0) 问题

Docker 容器运行后状态显示为 `Exited (0)` 的原因分析和解决方法：

### 1.1 问题分析

- **Exited (0)** 表示容器正常退出，没有错误，这通常是因为容器内的主进程已经完成了它的任务并退出了
- 可能的原因是容器内的入口点脚本或命令没有正确配置，导致容器在启动后立即退出

### 1.2 解决方法

- 检查 Dockerfile 中的 `ENTRYPOINT` 或 `CMD` 指令，确保它们指向一个持续运行的进程或服务
- 确保容器内的环境变量和工作目录设置正确，以便容器能够正常运行
- 如果需要容器持续运行，可以在 Dockerfile 中添加一个持续运行的命令，例如 `tail -f /dev/null`，或者确保主进程是一个守护进程

### 1.3 调试技巧

- 通过 `docker logs <container_id>` 查看容器的日志，以获取更多信息
- 如果容器需要运行 JupyterHub 或其他服务，确保相关的配置和依赖项已正确设置
- 可以尝试在容器内运行一个交互式 shell：`docker run -it <image_name> /bin/bash`，以便检查容器内的环境和配置
- 如果容器需要访问特定的端口或资源，确保在运行容器时正确映射了端口和挂载了必要的卷
## 2. 检查基础镜像配置

如果 Dockerfile 配置没有问题，需要检查 FROM 基础镜像的配置是否正确：

### 2.1 检查基础镜像配置方法

检查基础镜像的 ENTRYPOINT 和 CMD 配置是否正确，确保它们指向一个持续运行的进程或服务。首先使用 docker inspect 查看基础镜像的详细信息：

```bash
docker inspect registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.1-kuae1.3.0-s4000-py38 | grep -A 5 Entrypoint
docker inspect registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.1-kuae1.3.0-s4000-py38 | grep -A 5 Cmd
```

### 2.2 示例输出分析

如果没有输出，说明基础镜像没有设置 ENTRYPOINT 和 CMD。如果有输出，则需要检查输出内容是否是持续运行的进程或服务。

```json
{"Entrypoint": [
                "/workspace/check_status.sh"
            ],
            "OnBuild": null,
            "Labels": {}
        },
{"Cmd": [
                "/bin/bash"
            ],
            "Image": "registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.0-kuae1.3.0",
            "Volumes": null,
            "WorkingDir": "/workspace",}
```

这两个输出都不是持续运行的进程或服务，因此需要修改 Dockerfile，添加一个持续运行的命令。
## 3. 创建新镜像

使用 Dockerfile 创建一个新的镜像：

```bash
git clone https://github.com/MooreThreads/tutorial_on_musa.git
cd tutorial_on_musa
docker build -t my-jupyterhub .
```

## 4. 运行新镜像

```bash
docker run -d \
    --privileged \
    -p 8008:8000 \
    --name=vllm_mtt_service \
    -w /workspace \
    -v /data/mtt/:/data/mtt/ \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80G \
    my-jupyterhub
```

## 5. 检查容器状态

```bash
docker ps -a | grep vllm_mtt_service
```

如果容器状态是 `Up`，则表示容器正在运行。如果容器状态是 `Exited (0)`，则表示容器已经正常退出。

## 6. 如果不创建Dockerfile，直接使用基础镜像

如果不想创建 Dockerfile，直接使用基础镜像运行容器：

```bash
docker run -d \
    --privileged \
    -p 8008:8000 \
    --name=vllm_mtt_service \
    -w /workspace \
    -v /data/mtt/:/data/mtt/ \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80G \
    --entrypoint "tail" \
    registry.mthreads.com/mcconline/mtt-vllm-public:v0.2.1-kuae1.3.0-s4000-py38 \
    -f /dev/null
```