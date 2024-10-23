#### device/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(device LANGUAGES CXX)
#### 载入cmake的MUSA模块
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)
#### 编译musa代码时打印信息
set(MUSA_VERBOSE_BUILD ON)
#### 添加额外的mcc编译选项
set(MUSA_MCC_FLAGS --offload-arch=mp_22 -Werror)
#### 添加头文件路径
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#### 让.cpp后缀的文件能被当成MUSA代码从而使用mcc编译
#### 若是mu或者cu后缀则无需指定，会自动被识别成MUSA代码
set_source_files_properties(device_module2.cpp
      PROPERTIES
      MUSA_SOURCE_PROPERTY_FORMAT OBJ
  )
#### 添加动态库libdevice.so
musa_add_library(device SHARED
  device_module1.mu
  device_module2.cpp
  )
