## 查看驱动版本

```
# 查看到驱动版本具体日期
sudo clinfo|grep Driver 

# 可查看驱动大版本号
mthreads-gmi
```

## 驱动版本和镜像版本对应关系

<details>
<summary>Docker Image List</summary>

| Driver Version | GPU |Docker Image |
| ---- | --- | --- |
| **Driver Version 20241025** | S4000 | registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38 |
| **Driver Version 20241025** | S3000 | registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S3000-py38 |
| **Driver Version 20241025** | S80 | registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S80-py38 |

</details> 

**NOTE:** Python3.9和Python3.10请分别使用“py39”和“py310”替换上述镜像中“py38”.


```bash
# S4000 Python3.8
docker run -it --privileged --pull always --network=host --name=torch_musa_test -w /home/workspace -v /data:/data --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38 /bin/bash
```

# 容器环境检查
```
# 在容器内执行
cd examples_on_musa/setup_musa/check
bash test_musa.sh
```
