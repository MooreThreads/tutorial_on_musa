## 1. [稍微复杂的工程项目](https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/)

一个实际的项目，会有明确的组织结构，一般不会将device代码和host端的代码混合在一个代码文件中，否则不利于项目的维护。我们考虑一个典型的device代码和host代码分离的项目。

其中host端代码如下：

[main.cpp](main.cpp)

host端代码不包含任何device的代码，对GPU的使用是通过封装好的host端的函数调用完成的。这些封装的使用GPU进行计算的函数接口声明在头文件device_func.h中：

[device_func.h](include/device_func.h)

而对GPU进行计算的函数实现则统一放在另外一个部分，这个例子中是放在device目录中：

[device_func.mu](device/device_func.mu)

这样的工程目录，编译项目可以使用如下步骤：

```bash
mkdir build
mcc ./device/device_func.mu -fPIC -c -o ./build/device_func.o
c++ ./build/device_func.o -fPIC -shared -o ./build/libdevice.so
c++ main.cpp ./build/libdevice.so -I ./include -L /usr/local/musa/lib -lmusart -o ./build/main
./build/main
func
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
```

工程项目编译过程往往会产生许多过程文件，我们先创建build目录来存放编译过程和结果的输出。第二步使用mcc编译器，将device目录下的.mu代码文件编译，这个过程会编译代码文件里面的device端MUSA代码。第三步我们将这些编译好的GPU相关代码整理成库文件供后续使用，这里演示生成动态链接库，因此在这个步骤和上一个步骤要使用-fPIC参数指示编译时按照地址无关的方式处理。最后第四步，编译和链接host端的代码。上面的项目编译流程是规范的，是干净的。使用GPU的加速代码一定是包含MUSA代码的，因此将他们归集到一个部分，编译时用mcc进行编译，然后生成库文件，可以是静态库也可以是动态库。GPU函数提供的接口声明在头文件中供host端代码使用，而host代码的编写则如往常一样，包含接口头文件，直接调用接口，在生成可执行文件的链接阶段链接上GPU函数的库文件即可。这种项目的结构，对于一个从原本纯CPU的程序进行GPU加速扩展，是非常自然的，GPU加速库可以独立编写，客户端程序仅仅是将原本CPU函数的接口调用改成相同功能的GPU接口调用。
