cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME transpose)
project(${PROJECT_NAME} LANGUAGES CXX)

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Find CUDA version: ${CUDA_VERSION_STRING}")
    cuda_add_executable(${PROJECT_NAME}  src/matrix_transpose.cu)
    
else()
    message(STATUS "CUDA NOT FOUND, Use Moore Threads MUSA")
    # 使用MUSA模块(cu/mu)
    list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
    find_package(MUSA)
    musa_add_executable(${PROJECT_NAME} src/matrix_transpose.cu)
    
endif()


