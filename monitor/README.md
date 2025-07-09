## -d deviceId
## -i 刷新时间
## -n 记录次数

## 执行脚本命令
./monitor_gpu.sh -d 0 -i 1 -n 10

## 后台运行命令
nohup ./monitor_gpu.sh -d 0 -i 1 -n 10 > /dev/null 2>&1 &
tail -f gpu_monitor_log.txt
