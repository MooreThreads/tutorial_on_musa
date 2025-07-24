#!/bin/bash

# 用法: ./widedeep_test.sh [image_tar_path] [image_name]
# 示例1: ./widedeep_test.sh musa_image.tar # 加载镜像起容器测试
# 示例2：./widedeep_test.sh                # 直接使用默认镜像起容器测试

set -e

# 解析输入参数
IMAGE_TAR=$1
# 如果未提供镜像名，则使用默认镜像
# ⚠️注意：请根据实际情况修改默认镜像名!!!
IMAGE_NAME=${2:-xxxxxxxxx/mt-ai/musa-pytorch-debian-py310:v2.1.0-ph1-v1.1-ddk0711-bytetest_widedeep_v1}

# 容器名添加时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# ⚠️注意：请根据实际情况修改容器名!!!
CONTAINER_NAME="widedeep_test_${TIMESTAMP}"

# 可选加载镜像
if [[ -n "$IMAGE_TAR" ]]; then
    echo "==> 加载镜像: $IMAGE_TAR"
    docker load -i "$IMAGE_TAR"
fi

# 启动容器
echo "==> 启动容器: $CONTAINER_NAME"
docker run -w /home/test/ -idt \
  --privileged \
  --network=host \
  --name="$CONTAINER_NAME" \
  --env MTHREADS_VISIBLE_DEVICES=all \
  --shm-size=80g \
  "$IMAGE_NAME" \
  /bin/bash

# 执行测试脚本
# ⚠️注：多条执行命令在一个 docker exec 中执行!!!!
CMDS="
cd /home/test/widedeep
for B in 512 1024 2048 4096; do
  echo '===== Running widedeep.py --batch '\$B
  python widedeep.py --batch \$B || echo '[警告] Batch='\$B' 执行异常'
  sleep 2
done
"

docker exec "$CONTAINER_NAME" bash -c "$CMDS"


echo "✅ 所有测试完成。容器名: $CONTAINER_NAME"
