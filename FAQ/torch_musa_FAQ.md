#### 1. 计算库无法找到

**问题描述**: 在安装 torch_musa wheel 包后，` import torch_musa` 时报错 : ImportError: libmudnn.so.1: cannot open shared object file: No such file or directory

**解决方法**: 确认 `/usr/local/musa/lib/`目录下是否有该库，如果没有的话，需要安装该数学库；如果有的话，需要执行：

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/musa/lib
```



#### 2. 算子不支持

**问题描述**: 在python环境调用`torch.tril`方法报错: NotImplementedError: could not run 'aten::tril.out"with arguments from the 'musa’ backend.

**解决方法**: 上述问题说明MUSA 后端缺少该算子的实现. 此时需要我们手动实现 tril 算子。考虑计算逻辑的完备性，建议把 tril group 内所有调用规则对应的 C++算子全都实现。  

- 算子支持列表: [torch_musa/tools/ops_scanner/ops_list.md at main · MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa/blob/main/tools/ops_scanner/ops_list.md)

- 算子开发指导: [torch_musa/docs at main · MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa/tree/main/docs)(MooreThreads-Torch_MUSA-Developer-Guide-CN-vx.x.x.pdf)




#### 3. 使用一键编译脚本build.sh编译时报错 git failed. Is it installed， and are you in a Git repository directory?

**问题描述**: 使用压缩包下载torch_musa源码后使用编译脚本bash build.sh编译whl包并安装时，报错 pre commit.errors.FatalError：error： git failed. Is it installed， and are you in a Git repository directory?

**解决方法**: 请使用git clone方式获取代码
```shell
git clone https://github.com/MooreThreads/torch_musa.git
```


