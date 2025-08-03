#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define BLOCK_SIZE 32
#define N 1024
#define PAD 1

// 1. 原始版本（存在Bank Conflict）
__global__ void bankConflictKernel(float* out, const float* in) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    tile[threadIdx.y][threadIdx.x] = in[y * N + x]; // 按行写入
    __syncthreads();
    
    out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // 按列读取（冲突）
}

// 2. Padding优化版本
__global__ void paddingOptimizedKernel(float* out, const float* in) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + PAD];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    tile[threadIdx.y][threadIdx.x] = in[y * N + x]; // 按行写入
    __syncthreads();
    
    out[y * N + x] = tile[threadIdx.x][threadIdx.y]; // 按列读取（无冲突）
}

// 3. Swizzle优化版本（通过位运算重排索引）
__global__ void swizzleOptimizedKernel(float* out, const float* in) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Swizzle写入：重排线程索引
    int swizzled_y = (threadIdx.y * 5 + threadIdx.x) % BLOCK_SIZE; // 5是任意选择的素数
    tile[swizzled_y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    
    // Swizzle读取：对称重排
    int swizzled_x = (threadIdx.x * 5 + threadIdx.y) % BLOCK_SIZE;
    out[y * N + x] = tile[threadIdx.y][swizzled_x];
}

// 4. 数据重排优化版本（对角线存储）
__global__ void reorderingKernel(float* out, const float* in) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 数据重排存储：对角线模式
    int store_idx = (threadIdx.x + threadIdx.y) % BLOCK_SIZE;
    tile[threadIdx.y][store_idx] = in[y * N + x];
    __syncthreads();
    
    // 对称重排读取
    int load_idx = (threadIdx.x + threadIdx.y) % BLOCK_SIZE;
    out[y * N + x] = tile[threadIdx.x][load_idx];
}

// 初始化数据
void initData(float* ip, int size) {
    for (int i = 0; i < size; i++) 
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

// 性能测试
void runTest() {
    const int mem_size = N * N * sizeof(float);
    float *h_in = (float*)malloc(mem_size);
    float *h_out = (float*)malloc(mem_size);
    initData(h_in, N * N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, mem_size);
    cudaMalloc(&d_out, mem_size);
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / block.x, N / block.y);

    // 预热GPU
    bankConflictKernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();

    // 测试原始版本
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) 
        bankConflictKernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    printf("Original: %.5f ms\n", std::chrono::duration<double>(end - start).count() * 10);

    // 测试Padding版本
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) 
        paddingOptimizedKernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    printf("Padding: %.5f ms\n", std::chrono::duration<double>(end - start).count() * 10);

    // 测试Swizzle版本
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) 
        swizzleOptimizedKernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    printf("Swizzle: %.5f ms\n", std::chrono::duration<double>(end - start).count() * 10);

    // 测试数据重排版本
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
        reorderingKernel<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    printf("Reordering: %.5f ms\n", std::chrono::duration<double>(end - start).count() * 10);

    // 清理
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
}

int main() {
    runTest();
    cudaDeviceReset();
    return 0;
}

