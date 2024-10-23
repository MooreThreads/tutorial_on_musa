## 1. [使用MUSA模块构建含GPU代码的项目](https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/)

从上面章节可以看到，对于一个实际的c++项目工程，代码文件往往是多个的，一般完整的项目构建流程，对于每一个代码文件需要一条编译命令生成 .o目标文件。对于阶段性的每一个库文件目标，需要一条链接命令执行。对于最终的每一个可执行文件的生成也需要一条链接命令执行。并且库文件的生成依赖 .o文件，可执行文件又依赖库文件或者 .o文件，这要求以上的编译和链接命令需要按照某种合适的顺序执行。对于大项目而言，直接用编译链接命令来构建项目是非常繁琐的。因此诞生了make，ninja等构建项目的工具，这些构建工具是通过描述目标的依赖关系以及生成目标的命令来编织流程的。由于不同平台的编译命令有差别，且不同的构建工具如make或ninja的实现也有差别，于是又诞生了cmake用以统一这些。使用cmake可以仅编写同一套CMakeLists.txt，生成不同平台不同编译工具使用的编译脚本，如make使用的Makefile或者ninja使用的build.ninja。

我们进一步增加项目的复杂度，目录结构如本案例

其中host端代码如下

[main1.cpp](main1.cpp)

[main2.cpp](main2.cpp)

接口文件如下：

[device_module1.h](include/device_module1.h)

[device_module2.h](include/device_module2.h)

device端的代码文件有：

[device_module1.mu](device/device_module1.mu)

[device_module2.cpp](device/device_module2.cpp)

device代码使用到的通用模板kernel放在kernel.muh

[kernel.muh](device/include/kernel.muh)

项目的结构关系是，device端代码会提供三个函数接口，其中模块一实现2个函数，模块二实现了1个函数，这两个模块会打包成一个动态库libdevice.so供host端的程序使用，而host端的两个不同的主程序分别调用了这三个接口。我们接下来用cmake来完成这个项目的构建。

项目的根CMakeLists.txt内容如下：

[CMakeLists.txt](CMakeLists.txt)

常规的手段编译两个主程序，注意要指定接口的头文件目录，以及需要指定需链接的GPU库，这个GPU库是在device子模块中编译生成的。

项目子目录device是一个子项目，里面同样包含一个CMakeLists.txt用于构建GPU库：

[CMakeLists.txt](device/CMakeLists.txt)

一开始需要使用 find_package(MUSA REQUIRED)，来载入MUSA模块。这里有一点需要注意，由于该模块暂时未被cmake官方收录，仅安装在了MUSA Toolkit的安装目录中，因此需要在载入模块之前将模块的安装目录更新到cmake的MODULE搜索路径中: list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)。接下来就可以使用musa_add_library这个在MUSA模块定义的cmake函数宏来指定为这个项目添加一个库目标。里面输入的所有源文件都会为其挨个生成编译命令，最后生成链接命令来打包成库。若需要为编译时提供头文件路径，则使用musa_include_directories函数宏。需要注意的是，源文件列表中的代码文件，若后缀名是 .mu或者 .cu，则会自动被识别用mcc编译器编译。在这个例子中故意将device_module2这个代码文件的后缀名写成 .cpp，来模拟一个情况。正如[README.md](../0_example_mul/README.md)所说的，虽然代码文件后缀名是 .cpp但是里面却含有MUSA代码，编译命令需要使用mcc编译器，并且加上-x musa编译参数。在cmake中，对于 .cpp后缀文件同样按默认约定是当成常规的c++代码文件的，默认使用c++编译器。这里为了明确告知cmake这个文件包含MUSA代码，可以用set_source_files_properties(device_module2.cpp PROPERTIES MUSA_SOURCE_PROPERTY_FORMAT OBJ)来设置该代码文件的文件属性。这样cmake就会把这个代码文件等同于 .mu后缀来处理。

**请注意：device/CMakeLists/txt文件中 set(MUSA_MCC_FLAGS --offload-arch=mp_21 -Werror)需要根据具体设备进行修改**
```
# 对于s80/s3000, arch=mp_21
set(MUSA_MCC_FLAGS --offload-arch=mp_21 -Werror)

# 对于s4000, arch=mp_22
set(MUSA_MCC_FLAGS --offload-arch=mp_22 -Werror)
```

这个案例的编译命令和运行结果如下：

```bash
cmake -B build
cmake --build build
./build/main1
mod1_func1
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
./build/main2
mod1_func2
y[0] = 4
y[1] = 8
y[2] = 12
y[3] = 16
mod2_func3
y[0] = 6
y[1] = 12
y[2] = 18
y[3] = 24
```
