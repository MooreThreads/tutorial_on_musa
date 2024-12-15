#include <stdio.h> 
#include <iostream> 
#include <algorithm>

typedef float real;

const int NUM_REPEATS = 100;
const int N = 10000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
 
void timing(real *h_x, real *d_x);

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
    CHECK(musaMemset(d_x, 0, M));
 
    printf("Using global memory only:\n");
    timing(h_x, d_x);
 
    free(h_x);
    CHECK(musaFree(d_x));
    return 0;
}
 
void __global__ reduce_global(real *d_x, real *d_y, const int n) {
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;
 
    unsigned int max_idx = n - blockDim.x * blockIdx.x;
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if ((tid + offset) < max_idx) {
                x[tid] += x[tid + offset];
            }
        }
        __syncthreads();
    }
 
    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}
 
real reduce(real *d_x) {
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(musaMalloc(&d_y, ymem));
    CHECK(musaMemset(d_y, 0, ymem));
    real *h_y = (real *)malloc(ymem);
 
    reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, N);
 
    CHECK(musaMemcpy(h_y, d_y, ymem, musaMemcpyDeviceToHost));
    
 
    real result = 0.0;
    for (int n = 0; n < grid_size; ++n) {
        result += h_y[n];
    }
 
    free(h_y);
    CHECK(musaFree(d_y));
    return result;
}
 
void timing(real *h_x, real *d_x) {
    real sum = 0;
 
    float total_time = 0.0f;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
        CHECK(musaMemcpy(d_x, h_x, M, musaMemcpyHostToDevice));
 
        musaEvent_t start, stop;
        CHECK(musaEventCreate(&start));
        CHECK(musaEventCreate(&stop));
        CHECK(musaEventRecord(start));
 
        sum = reduce(d_x);
 
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
