#### CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(musa_fft_example LANGUAGES CXX)
## 载入MUSAToolkit模块
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSAToolkit REQUIRED)
## 添加可执行文件
add_executable(main main.cpp)
## 为目标链接运行时库及数学库
target_link_libraries(main PRIVATE
  MUSA::musart
  MUSA::mufft
  )