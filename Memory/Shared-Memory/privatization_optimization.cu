#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// 非私有化版本 - 存在数据竞争和同步开销
__global__ void non_privatized_kernel(float* input, float* output, float* global_sum, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 共享变量 - 所有线程访问同一位置，存在数据竞争
    float local_sum = 0.0f;
    
    // 模拟复杂的计算
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // 复杂的数学运算
        local_sum += sinf(input[i]) * cosf(input[i]) + sqrtf(fabsf(input[i]));
    }
    
    // 原子操作更新全局和 - 高昂的同步开销
    atomicAdd(global_sum, local_sum);
    
    // 可能的写入操作
    if(tid < n) {
        output[tid] = local_sum;
    }
}

// 私有化优化版本 - 每个线程拥有自己的私有变量
__global__ void privatized_kernel(float* input, float* output, float* global_sum, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 私有变量 - 每个线程独立拥有
    float local_sum = 0.0f;
    
    // 模拟复杂的计算 - 完全私有化
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // 复杂的数学运算
        local_sum += sinf(input[i]) * cosf(input[i]) + sqrtf(fabsf(input[i]));
    }
    
    // 在局部完成操作后，只在必要时进行原子操作
    // 或者使用共享内存进行块内聚合
    if(tid == 0) {
        atomicAdd(global_sum, local_sum);
    }
    
    // 每个线程独立处理自己的输出
    if(tid < n) {
        output[tid] = local_sum;
    }
}

// 使用共享内存的私有化优化版本 - 进一步优化
__global__ void privatized_with_shared_kernel(float* input, float* output, float* global_sum, int n) {
    extern __shared__ float shared_data[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // 每个线程独立计算
    float local_sum = 0.0f;
    
    // 模拟复杂的计算
    for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
        local_sum += sinf(input[i]) * cosf(input[i]) + sqrtf(fabsf(input[i]));
    }
    
    // 将局部结果存储到共享内存
    shared_data[local_tid] = local_sum;
    __syncthreads();
    
    // 块内聚合
    if(local_tid == 0) {
        float block_sum = 0.0f;
        for(int i = 0; i < blockDim.x; i++) {
            block_sum += shared_data[i];
        }
        atomicAdd(global_sum, block_sum);
    }
    
    // 每个线程独立处理自己的输出
    if(tid < n) {
        output[tid] = local_sum;
    }
}

// 初始化数据
void initData(float* data, int size) {
    for(int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 1000) / 100.0f;
    }
}

// 性能测试函数
void runTest() {
    const int N = 10000000;  // 数据量
    const int blockSize = 256;
    const int gridSize = 512;  // 更多线程
    const int iterations = 100;  // 迭代次数以获得稳定的时间测量
    
    size_t dataSize = N * sizeof(float);
    
    // 分配主机内存
    float* h_input = (float*)malloc(dataSize);
    float* h_output1 = (float*)malloc(dataSize);
    float* h_output2 = (float*)malloc(dataSize);
    float* h_output3 = (float*)malloc(dataSize);
    
    // 初始化输入数据
    initData(h_input, N);
    
    // 分配设备内存
    float *d_input, *d_output1, *d_output2, *d_output3, *d_global_sum1, *d_global_sum2, *d_global_sum3;
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output1, dataSize);
    cudaMalloc(&d_output2, dataSize);
    cudaMalloc(&d_output3, dataSize);
    cudaMalloc(&d_global_sum1, sizeof(float));
    cudaMalloc(&d_global_sum2, sizeof(float));
    cudaMalloc(&d_global_sum3, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_global_sum1, 0, sizeof(float));
    cudaMemset(d_global_sum2, 0, sizeof(float));
    cudaMemset(d_global_sum3, 0, sizeof(float));
    
    // 预热GPU
    non_privatized_kernel<<<gridSize, blockSize>>>(d_input, d_output1, d_global_sum1, N);
    cudaDeviceSynchronize();
    
    // 测试非私有化版本
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        non_privatized_kernel<<<gridSize, blockSize>>>(d_input, d_output1, d_global_sum1, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double nonPrivatizedTime = std::chrono::duration<double>(end - start).count() * 1000;
    printf("Non-privatized version: %.5f ms\n", nonPrivatizedTime);
    
    // 测试私有化版本
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        privatized_kernel<<<gridSize, blockSize>>>(d_input, d_output2, d_global_sum2, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double privatizedTime = std::chrono::duration<double>(end - start).count() * 1000;
    printf("Privatized version: %.5f ms\n", privatizedTime);
    
    // 测试私有化+共享内存版本
    int sharedMemSize = blockSize * sizeof(float);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        privatized_with_shared_kernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_output3, d_global_sum3, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double sharedPrivatizedTime = std::chrono::duration<double>(end - start).count() * 1000;
    printf("Privatized with shared memory version: %.5f ms\n", sharedPrivatizedTime);
    
    // 计算加速比
    if(nonPrivatizedTime > 0) {
        double speedup1 = nonPrivatizedTime / privatizedTime;
        double speedup2 = nonPrivatizedTime / sharedPrivatizedTime;
        printf("\nSpeedup comparisons:\n");
        printf("Privatized vs Non-privatized: %.2fx\n", speedup1);
        printf("Shared Privatized vs Non-privatized: %.2fx\n", speedup2);
    }
    
    // 验证结果
    cudaMemcpy(h_output1, d_output1, dataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, dataSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output3, d_output3, dataSize, cudaMemcpyDeviceToHost);
    
    // 简单验证：比较前几个元素
    bool correct = true;
    for(int i = 0; i < 10 && correct; i++) {
        if(fabsf(h_output1[i] - h_output2[i]) > 1e-3) {
            printf("Verification failed at index %d: %.6f vs %.6f\n", i, h_output1[i], h_output2[i]);
            correct = false;
        }
    }
    
    if(correct) {
        printf("Result verification: PASSED\n");
    } else {
        printf("Result verification: FAILED\n");
    }
    
    // 清理内存
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_global_sum1);
    cudaFree(d_global_sum2);
    cudaFree(d_global_sum3);
}

int main() {
    printf("=== GPU Privatization Optimization ===\n");
    printf("Processing %d elements with complex mathematical operations\n", 10000000);
    runTest();
    cudaDeviceReset();
    return 0;
}