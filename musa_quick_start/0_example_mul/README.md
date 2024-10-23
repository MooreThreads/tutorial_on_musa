## 1. [简单的MUSA程序](https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/)

我们从最简单的MUSA程序说起，项目目录如下，仅有一个mu代码文件。

[main.mu](main.mu)

编译代码只需要执行简单编译命令即可，编译和执行结果如下：

```bash
mcc main.mu -lmusart -o main
./main
func
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
```

main.mu 文件算是最小的MUSA代码，里面主函数执行了GPU程序的典型步骤：

申请显存， 将数据从host传输到GPU上， 执行device函数进行计算， 从GPU将数据取回host， 释放显存。 其中展示的device上的计算任务是简单的向量缩放，对输入数据的每个元素乘以一个常数。代码文件后缀名为 .mu，编译器mcc会识别这个后缀名并以此为依据认为代码文件中包含device代码的定义和调用，即__global__前缀的函数定义，和主函数中三尖括号<<<...>>>标记的函数调用，这两个是MUSA代码的最主要的标志，只能使用mcc编译器编译。

倘若代码文件命名为main.cpp，即后缀为 .cpp，那么用上面的命令编译将会报错。原因是 .cpp后缀默认约定指示该代码文件是常规的c++代码并不包含device函数，这将自动调用host端的编译器如g++执行编译。于是g++将无法识别MUSA代码的语法而报错。这个时候需要执行的编译命令是

```bash
mcc -x musa main.cpp -lmusart -o main
```

其中需要在代码文件main.cpp的前面添加编译参数 -x musa, 这个编译参数告诉mcc，虽然这个文件后缀名是 .cpp但是它里面的内容是包含MUSA代码的，需要用mcc来执行编译。
