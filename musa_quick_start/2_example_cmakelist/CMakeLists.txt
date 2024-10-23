#### CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(2_example_cmakelist LANGUAGES CXX)
#### 在子项目中编译libdevice库，供后面链接使用
add_subdirectory(device)
#### 编译程序1
add_executable(main1 main1.cpp)
target_include_directories(main1 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
target_link_libraries(main1 PRIVATE device)
#### 编译程序2
add_executable(main2 main2.cpp)
target_include_directories(main2 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
target_link_libraries(main2 PRIVATE device)
