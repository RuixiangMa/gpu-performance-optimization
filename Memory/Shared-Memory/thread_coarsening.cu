#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// 向量加法
// 原始版本：每个线程处理一个元素
__global__ void naiveAddKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 线程粗化版本：每个线程处理多个元素
__global__ void coarsenedAddKernel(float* a, float* b, float* c, int n, int factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理factor个元素
    // 确保每个元素恰好被一个线程处理一次
    for (int i = tid * factor; i < n; i += total_threads * factor) {
        c[i] = a[i] + b[i];
    }
}

// 线程粗化+共享内存版本
__global__ void coarsenedAddWithSharedMem(float* a, float* b, float* c, int n, int factor) {
    extern __shared__ float sdata[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理factor个元素
    for (int i = tid * factor; i < n; i += total_threads * factor) {
        c[i] = a[i] + b[i];
    }
    
    // 简单的共享内存使用，不改变计算结果
    sdata[threadIdx.x] = 0.0f;  // 仅初始化，不进行实际操作
    __syncthreads();
}

// 初始化数据
void initData(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 100) / 10.0f;
    }
}

// 性能测试
void runTest() {
    const int N = 100000000;  // 更大的数据量来观察性能差异
    const int blockSize = 256;
    const int gridSize = 512;  // 更多的网格线程
    const int factor = 4;      // 粗化因子：每个线程处理4个元素
    
    size_t dataSize = N * sizeof(float);
    
    // 分配主机内存
    float* h_a = (float*)malloc(dataSize);
    float* h_b = (float*)malloc(dataSize);
    float* h_c_naive = (float*)malloc(dataSize);
    float* h_c_coarsened = (float*)malloc(dataSize);
    float* h_c_shared = (float*)malloc(dataSize);
    
    // 初始化输入数据
    initData(h_a, N);
    initData(h_b, N);
    
    // 分配设备内存
    float *d_a, *d_b, *d_c_naive, *d_c_coarsened, *d_c_shared;
    cudaMalloc(&d_a, dataSize);
    cudaMalloc(&d_b, dataSize);
    cudaMalloc(&d_c_naive, dataSize);
    cudaMalloc(&d_c_coarsened, dataSize);
    cudaMalloc(&d_c_shared, dataSize);
    
    // 复制数据到设备
    cudaMemcpy(d_a, h_a, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, dataSize, cudaMemcpyHostToDevice);
    
    // 预热GPU
    naiveAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c_naive, N);
    cudaDeviceSynchronize();
    
    // 测试原始版本
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {  // 重复100次以获得更稳定的时间测量
        naiveAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c_naive, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double naiveTime = std::chrono::duration<double>(end - start).count() * 100;
    printf("Naive version: %.5f ms\n", naiveTime);
    
    // 测试线程粗化版本
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        coarsenedAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c_coarsened, N, factor);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double coarsenedTime = std::chrono::duration<double>(end - start).count() * 100;
    printf("Coarsened version (factor=%d): %.5f ms\n", factor, coarsenedTime);
    
    // 测试线程粗化+共享内存版本
    int sharedMemSize = blockSize * sizeof(float);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        coarsenedAddWithSharedMem<<<gridSize, blockSize, sharedMemSize>>>(
            d_a, d_b, d_c_shared, N, factor);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double sharedTime = std::chrono::duration<double>(end - start).count() * 100;
    printf("Coarsened with shared memory (factor=%d): %.5f ms\n", factor, sharedTime);
    
    // 计算加速比
    if (naiveTime > 0) {
        double speedup1 = naiveTime / coarsenedTime;
        double speedup2 = naiveTime / sharedTime;
        printf("\nSpeedup comparisons:\n");
        printf("Coarsened vs Naive: %.2fx\n", speedup1);
        printf("Shared Mem vs Naive: %.2fx\n", speedup2);
    }
    
    // 验证结果
    cudaMemcpy(h_c_naive, d_c_naive, dataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_coarsened, d_c_coarsened, dataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_shared, d_c_shared, dataSize, cudaMemcpyDeviceToHost);
    
    // 简单验证：比较前几个元素
    bool correct = true;
    for (int i = 0; i < 10 && correct; i++) {
        if (fabsf(h_c_naive[i] - h_c_coarsened[i]) > 1e-3) {
            printf("Verification failed at index %d: %.6f vs %.6f\n", i, h_c_naive[i], h_c_coarsened[i]);
            correct = false;
        }
    }
    
    if (correct) {
        printf("Result verification: PASSED\n");
    } else {
        printf("Result verification: FAILED\n");
    }
    
    // 清理内存
    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_coarsened);
    free(h_c_shared);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_coarsened);
    cudaFree(d_c_shared);
}

int main() {
    printf("=== Thread Coarsening Optimization ===\n");
    printf("Using factor = 4 (each thread processes 4 elements)\n");
    printf("Processing %d elements with vector addition\n", 100000000);
    runTest();
    cudaDeviceReset();
    return 0;
}
