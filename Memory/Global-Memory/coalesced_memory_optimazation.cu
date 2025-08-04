#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int TILE_DIM = 32;
const int MATRIX_SIZE = 1024;

// 1. 基础转置（非合并写入）
__global__ void naiveTranspose(float* out, const float* in) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        out[x * MATRIX_SIZE + y] = in[y * MATRIX_SIZE + x]; // 写入时非合并
    }
}

// 2. 优化转置（合并写入+共享内存）
__global__ void optimizedTranspose(float* out, const float* in) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1避免bank conflict
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 合并读取全局内存
    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        tile[threadIdx.y][threadIdx.x] = in[y * MATRIX_SIZE + x];
    }
    __syncthreads();
    
    // 转置写入（交换x/y保证合并）
    x = blockIdx.y * TILE_DIM + threadIdx.x; // 注意blockIdx.y
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        out[y * MATRIX_SIZE + x] = tile[threadIdx.x][threadIdx.y]; // 合并写入
    }
}

void benchmark(float* d_out, float* d_in, const char* name, 
               void (*kernel)(float*, const float*)) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((MATRIX_SIZE + TILE_DIM - 1) / TILE_DIM, 
              (MATRIX_SIZE + TILE_DIM - 1) / TILE_DIM);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        kernel<<<grid, block>>>(d_out, d_in);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms/iter\n", name, ms / 100);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = MATRIX_SIZE * MATRIX_SIZE;
    float *h_in = new float[N];
    float *h_out = new float[N];
    
    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_in[i] = i * 1.0f;
    }
    
    // 设备内存分配
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 性能对比
    benchmark(d_out, d_in, "Naive Transpose (Uncoalesced)", naiveTranspose);
    benchmark(d_out, d_in, "Optimized Transpose (Coalesced)", optimizedTranspose);
    
    // 验证结果
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int y = 0; y < 10 && correct; ++y) {
        for (int x = 0; x < 10; ++x) {
            if (h_out[x * MATRIX_SIZE + y] != h_in[y * MATRIX_SIZE + x]) {
                printf("Mismatch at (%d,%d): %.1f != %.1f\n", 
                       x, y, h_out[x*MATRIX_SIZE+y], h_in[y*MATRIX_SIZE+x]);
                correct = false;
                break;
            }
        }
    }
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");
    
    // 清理
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}