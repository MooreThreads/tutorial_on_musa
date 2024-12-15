#pragma once
#include <stdio.h>

// CHECK为宏函数名，call为一个runtime api

#define CHECK(call)                                                        \
    do                                                                     \
    {                                                                      \
        const cudaError_t error_code = call;                               \
        if (error_code != cudaSuccess)                                     \
        {                                                                  \
            printf("    Error:\n");                                        \
            printf("    File:      %s\n", __FILE__);                       \
            printf("    Line:      %d\n", __LINE__);                       \
            printf("    Error code:%d\n", error_code);                     \
            printf("    Error text:%s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                       \
        }                                                                  \
    } while (0);
