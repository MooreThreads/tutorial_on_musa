#include <stdio.h> 
#include <iostream> 
#include <algorithm>

typedef float real;
 
const int NUM_REPEATS = 100;
const int N = 10000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
 
void timing(real *h_x, real *d_x, const int method);
 
#define CHECK(cmd) \
    if (musaSuccess != cmd) { \
        fprintf(stderr, "error code %s!\n", musaGetErrorString(musaGetLastError())); \
        exit(EXIT_FAILURE); \
    } 

int main(void) {
    real *h_x = (real *)malloc(M);
    for (int n = 0; n < N; ++n) {
        h_x[n] = 1.01;
    }
    real *d_x;
    CHECK(musaMalloc(&d_x, M));
 
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 1);
 
    free(h_x);
    CHECK(musaFree(d_x));
    return 0;
}
 
void __global__ reduce_shared(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();
 
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
 
    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}
 
void __global__ reduce_dynamic(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();
 
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
 
    if (tid == 0) {
        d_y[bid] = s_y[0];
    }
}
 
real reduce(real *d_x, const int method) {
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(musaMalloc(&d_y, ymem));
    real *h_y = (real *)malloc(ymem);
 
    switch (method) {
        case 0:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }
 
    CHECK(musaMemcpy(h_y, d_y, ymem, musaMemcpyDeviceToHost));
 
    real result = 0.0;
    for (int n = 0; n < grid_size; ++n) {
        result += h_y[n];
    }
 
    free(h_y);
    CHECK(musaFree(d_y));
    return result;
}
 
void timing(real *h_x, real *d_x, const int method) {
    real sum = 0;
 
    float total_time;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
        CHECK(musaMemcpy(d_x, h_x, M, musaMemcpyHostToDevice));
 
        musaEvent_t start, stop;
        CHECK(musaEventCreate(&start));
        CHECK(musaEventCreate(&stop));
        CHECK(musaEventRecord(start));
        musaEventQuery(start);
 
        sum = reduce(d_x, method);
 
        CHECK(musaEventRecord(stop));
        CHECK(musaEventSynchronize(stop));
        float elapsed_time;
        CHECK(musaEventElapsedTime(&elapsed_time, start, stop));
 
        if (repeat > 0) {
            total_time += elapsed_time;
        }
 
        CHECK(musaEventDestroy(start));
        CHECK(musaEventDestroy(stop));
    }
    printf("average time = %f.\n", total_time / NUM_REPEATS);
 
    printf("sum = %f.\n", sum);
}
