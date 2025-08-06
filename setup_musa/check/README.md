# ib_stream_test.sh
这个脚本主要是用来测试双机IB设备之间的打流，脚本实现的原理很简单：
1. 双机ssh免密
2. A机器固定一个接收端，B机器遍历所有IB设备进行打流，
3. A机器进入下一个IB设备进行接收，重复上面的动作直到所有IB设备组合都测试完毕


### 补充手动打流方法：
1. 首先查看ib设备与网卡名对应，以及状态
```shell
ibdev2netdev
```
2. 以ib_write_bw命令为例测试打流，所有的网卡全部需要测测试，这里仅以其中一个为例
```shell
# server端，业务IP 10.2.38.10，例如测试mlx5_9
ib_write_bw -d mlx5_9

# clinet端，业务IP 10.2.38.11
ib_write_bw -d mlx5_9 --report_gbits 10.2.38.10
