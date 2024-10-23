## 1. [使用MUSAToolkit模块构建项目](https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/)

cmake版本3.17之后，官方新增加了CUDAToolkit模块。其用途是，有些工程并没有包含任何GPU端的kernel代码，但是调用了NVIDIA官方提供的数学库或图像处理库等现成的c或c++接口，从而使用GPU进行加速。这种情况下整个工程内全部代码都是c或c++语言编写，无任何CUDA代码，故无需使用nvcc编译器编译工程。那么整个项目仅需要使用host端的c或c++编译器编译，最后链接的时候把运行时库以及数学库等添加上即可。CUDAToolkit模块则提供了所有使用GPU加速可能使用到的库目标。为了保证对CUDA使用的最佳兼容，MUSAToolkit中也包含了cmake的MUSAToolkit模块。

例如本案例项目：

其中主函数里面是使用mufft数学库进行傅里叶变化计算：

[main.cpp](main.cpp)

该项目的CMakeListst.txt如下：

[CMakeLists.txt](CMakeLists.txt)

在一开始需要用find_package(MUSAToolkit REQUIRED)，来载入MUSAToolkit模块。同样的由于该模块暂时未被cmake官方收录，仅安装在了MUSA Toolkit的安装目录中，因此需要在载入模块之前将模块的安装目录更新到cmake的MODULE搜索路径中: list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)。

模块载入之后，将提供若干库目标以及变量供使用，这个例子中用到了运行时库和傅里叶变换库，故给目标添加链接库MUSA::musart和MUSA::mufft。需要指出的是，这里模块提供的目标MUSA::已经包含了所需的头文件路径，会自动传递给要编译的目标，故无需再给编译目标添加MUSA相关的头文件目录。

这个案例的编译命令和运行结果如下：

```bash
$ cmake -B build
$ cmake --build build
$ ./build/main
(28, 28)
(-13.6569, 5.65685)
(-8, 0)
(-5.65685, -2.34315)
(-4, -4)
(-2.34315, -5.65685)
(0, -8)
(5.65685, -13.6569)
```

## 2. 总结

MUSA沿用了Modules的方式，也提供了相似的cmake模块供使用。保留了和CUDA几乎完全一致的使用方式，以达到用户尽可能方便地构建MUSA工程。这个兼容性也能带来快速迁移CUDA项目的便捷。在做项目迁移时，若项目使用cmake工具构建，则绝大多数情况下可以仅做文本替换，将CMakelist.txt中的CUDA替换成MUSA，CU前缀替换成MU前缀。
