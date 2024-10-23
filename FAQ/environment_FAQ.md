#### 1. kernel 版本不匹配问题

**问题描述**: 执行 `sudo apt install libgbm1 libglapi-mesa`  安装依赖过程中报错: Error! Bad return status for module build on kernel: 6.5.0-41-generic (x86 64)

**解决方法**: 若内核版本过高，可能导致依赖包装不上去, 官方文档给出内核版本范围：`5.4.X-5.15.X`, 请将kernel版本降级.  参考[kernel_install.md](../docs/other/kernel_install.md)



#### 2. 依赖包嵌套安装

**问题描述**: 使用命令`apt install lightdm`时报错: E:Unmet dependencies.Try 'apt --fix-broken install' with no packages (or specify a solution).

**解决方法**: 首先执行命令`sudo apt update`更新软件源数据库, 然后执行命令`sudo apt --fix-broken install`，用于尝试自动修复由于依赖关系不满足而导致的软件包管理问题



#### 3. IOMMU未开启导致推理异常

**问题描述**: host环境检验没有问题, 执行`mthreads-gmi`正常输出, 但是在容器中做模型推理报错: 

```shell
/opt/conda/envs/py38/lib/python3.8/site-packages/torch_musa/core/device.py:156: UserWarning: MUSA initialization: Unexpected error from musaGetDeviceCount(). Did you run some musa functions before calling NumCudaDevices() that might have already set an error? Error 801: operation not supported (Triggered internally at /home/torch musa/torch musa/csrc/core/Device.cpp:93.)
 return torch_musa._MUSAC_._musa_getDeviceCount()
[INF0] device count 0
Traceback(most recent call last):
  File "generate_demo.py",line 5,in <module>
  	model = mttransformer.LLMEngine(args)
  File "/opt/conda/envs/py38/lib/python3.8/site-packages/mttransformer/llm_engine.py", line 138, in __init__
assert device_count>0
AssertionError
```

**解决方法**: IOMMU未开启导致, 需要开启IOMMU, 开启步骤如下:

```python
# 如果你的 CPU 是 AMD/HG, 请改为 amd_iommu=on, 而不是 intel_iommu=on
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu.passthrough=0"/'
/etc/default/grub
# 如果你的 CPU 是 AMD/HG, 请改为 amd_iommu=on, 而不是 intel_iommu=on
sudo update-grub
sudo reboot
# 根据如下命令判断是否开启 IOMMU
sudo cat /var/log/dmesg | grep -e "AMD-Vi: Interrupt remapping enabled" -e "IOMMU enabled"
```



#### 4. 容器内 mthreads-gmi 无输出

**问题描述**: host环境中执行`mthreads-gmi`正常输出, 但是在容器当中执行`mthreads-gmi`无输出

**解决方法**: 将host环境中的mthreads-gmi文件cp到容器中mthreads-gmi对应位置. 可使用命令`which mthreads-gmi`查看.



#### 5. 显示GPU显存不足问题

**问题描述**: 以S4000卡为例, 单卡49152MiB显存(48G), `mthreads-gmi` 输出单卡显存为32768MiB(32G)

```shell
Tue Nov 12 20:55:00 2024
---------------------------------------------------------------
    mthreads-gmi:1.12.2          Driver Version:1.2.0
---------------------------------------------------------------
ID   Name           |PCIe                |%GPU  Mem
     Device Type    |Pcie Lane Width     |Temp  MPC Capable
                                         |      ECC Mode
+-------------------------------------------------------------+
0    MTT S4000      |00000000:08:00.0    |88%   44302MiB(32768MiB)
     Physical       |16x(16x)            |65C   YES
                                         |      N/A
```

**解决方法**: 为IOMMU问题, 如果未开启则需要开启, 如果开启, 则未生效, 需重启IOMMU. 开启方法见 **问题3**



#### 6. mt-container-toolkit未成功安装

**问题描述**: 在 docker container 内部使用 torch_musa 时，报错 ImportError: libsrv_um_MUSA.so: cannot open shared object file: No such file or directory 或者 ImportError: /usr/lib/x86_64-linux-gnu/musa/libsrv_um_MUSA.so: file too short ？

**解决方法**: mt-container-toolkit 未安装或者安装之后未绑定摩尔线程容器运行时到 Docker. 详情参考: [安装指导 | 摩尔线程文档中心](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/install_guide)