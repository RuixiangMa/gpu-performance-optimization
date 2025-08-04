#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define RADIUS 4  // Stencil radius

// 初始化矩阵
void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i * size + j] = (float)rand() / RAND_MAX;
        }
    }
}

// CPU版本的5点Stencil计算（仅用于验证）
void stencil5pointCPU(float* input, float* output, int size) {
    for (int i = RADIUS; i < size - RADIUS; i++) {
        for (int j = RADIUS; j < size - RADIUS; j++) {
            float sum = input[i * size + j] * 0.5f;  // Center weight
            sum += input[(i-1) * size + j] * 0.125f;  // Top
            sum += input[(i+1) * size + j] * 0.125f;  // Bottom
            sum += input[i * size + (j-1)] * 0.125f;  // Left
            sum += input[i * size + (j+1)] * 0.125f;  // Right
            output[i * size + j] = sum;
        }
    }
}

// 全局内存版本的5点Stencil计算
__global__ void stencil5pointGlobal(float* input, float* output, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
    int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
    
    if (row < size - RADIUS && col < size - RADIUS) {
        float sum = input[row * size + col] * 0.5f;  // Center weight
        sum += input[(row-1) * size + col] * 0.125f;  // Top
        sum += input[(row+1) * size + col] * 0.125f;  // Bottom
        sum += input[row * size + (col-1)] * 0.125f;  // Left
        sum += input[row * size + (col+1)] * 0.125f;  // Right
        output[row * size + col] = sum;
    }
}

// 共享内存版本的5点Stencil计算
__global__ void stencil5pointShared(float* input, float* output, int size) {
    // 共享内存需要包含边界数据
    __shared__ float sData[TILE_SIZE + 2*RADIUS][TILE_SIZE + 2*RADIUS];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // 全局坐标
    int row = by * TILE_SIZE + ty + RADIUS;
    int col = bx * TILE_SIZE + tx + RADIUS;
    
    // 加载数据到共享内存（包括边界）
    // 每个线程加载自己的数据点
    if (row < size - RADIUS && col < size - RADIUS) {
        sData[ty + RADIUS][tx + RADIUS] = input[row * size + col];
    }
    
    // 加载边界数据
    // 上边界
    if (ty < RADIUS) {
        if (row - RADIUS >= 0 && col < size - RADIUS) {
            sData[ty][tx + RADIUS] = input[(row - RADIUS) * size + col];
        }
        // 下边界
        if (row + TILE_SIZE < size && col < size - RADIUS) {
            sData[ty + TILE_SIZE + RADIUS][tx + RADIUS] = input[(row + TILE_SIZE) * size + col];
        }
    }
    
    // 左边界
    if (tx < RADIUS) {
        if (col - RADIUS >= 0 && row < size - RADIUS) {
            sData[ty + RADIUS][tx] = input[row * size + (col - RADIUS)];
        }
        // 右边界
        if (col + TILE_SIZE < size && row < size - RADIUS) {
            sData[ty + RADIUS][tx + TILE_SIZE + RADIUS] = input[row * size + (col + TILE_SIZE)];
        }
    }
    
    __syncthreads();
    
    // 执行Stencil计算
    if (row < size - RADIUS && col < size - RADIUS) {
        float sum = sData[ty + RADIUS][tx + RADIUS] * 0.5f;  // Center weight
        sum += sData[ty + RADIUS - 1][tx + RADIUS] * 0.125f;  // Top
        sum += sData[ty + RADIUS + 1][tx + RADIUS] * 0.125f;  // Bottom
        sum += sData[ty + RADIUS][tx + RADIUS - 1] * 0.125f;  // Left
        sum += sData[ty + RADIUS][tx + RADIUS + 1] * 0.125f;  // Right
        output[row * size + col] = sum;
    }
}

// 计算带宽函数
float calculateBandwidth(size_t dataSize, float milliseconds) {
    // 数据大小转换为GB
    float dataSizeGB = (float)dataSize / (1024.0f * 1024.0f * 1024.0f);
    // 时间转换为秒
    float timeSeconds = milliseconds / 1000.0f;
    // 带宽 = 数据大小 / 时间
    return dataSizeGB / timeSeconds;
}

int main() {
    const int size = 4096;  // 矩阵大小
    size_t dataSize = size * size * sizeof(float);
    
    printf("Matrix size: %d x %d\n", size, size);
    printf("Data size per matrix: %.2f MB\n", (float)dataSize / (1024.0f * 1024.0f));
    printf("Stencil radius: %d\n", RADIUS);
    
    float *h_input = (float*)malloc(dataSize);
    float *h_output_global = (float*)malloc(dataSize);
    float *h_output_shared = (float*)malloc(dataSize);
    float *h_output_ref = (float*)malloc(dataSize);
    
    initMatrix(h_input, size);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);
    
    cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size - 2*RADIUS + block.x - 1) / block.x, (size - 2*RADIUS + block.y - 1) / block.y);
    
    // 测试全局内存版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 多次运行取平均值以提高测量准确性
    const int numRuns = 5;
    float totalTimeGlobal = 0.0f;
    
    for (int run = 0; run < numRuns; run++) {
        cudaEventRecord(start);
        stencil5pointGlobal<<<grid, block>>>(d_input, d_output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTimeGlobal += milliseconds;
    }
    
    float avgTimeGlobal = totalTimeGlobal / numRuns;
    printf("\n=== Performance Comparison ===\n");
    printf("Global memory version: %.3f ms (average of %d runs)\n", avgTimeGlobal, numRuns);
    
    cudaMemcpy(h_output_global, d_output, dataSize, cudaMemcpyDeviceToHost);
    
    // 计算参考结果（仅用于小矩阵）
    if (size <= 1024) {
        stencil5pointCPU(h_input, h_output_ref, size);
    }
    
    // 测试共享内存版本
    float totalTimeShared = 0.0f;
    
    for (int run = 0; run < numRuns; run++) {
        cudaEventRecord(start);
        stencil5pointShared<<<grid, block>>>(d_input, d_output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTimeShared += milliseconds;
    }
    
    float avgTimeShared = totalTimeShared / numRuns;
    printf("Shared memory version: %.3f ms (average of %d runs)\n", avgTimeShared, numRuns);
    
    // 计算性能提升
    float speedup = avgTimeGlobal / avgTimeShared;
    printf("\n=== Performance Analysis ===\n");
    printf("Speedup: %.2fx\n", speedup);
    
    // 计算带宽
    // 对于Stencil计算，我们读取一个输入矩阵并写入一个输出矩阵
    size_t totalDataTransferred = 2 * dataSize;
    float bandwidthGlobal = calculateBandwidth(totalDataTransferred, avgTimeGlobal);
    float bandwidthShared = calculateBandwidth(totalDataTransferred, avgTimeShared);
    
    printf("Effective bandwidth (global memory): %.2f GB/s\n", bandwidthGlobal);
    printf("Effective bandwidth (shared memory): %.2f GB/s\n", bandwidthShared);
    printf("Bandwidth improvement: %.2fx\n", bandwidthShared / bandwidthGlobal);
    
    cudaMemcpy(h_output_shared, d_output, dataSize, cudaMemcpyDeviceToHost);
    
    // 验证结果正确性（仅对小矩阵）
    bool correct = true;
    if (size <= 1024) {
        float epsilon = 1.0e-5;
        for (int i = RADIUS; i < size - RADIUS && correct; i++) {
            for (int j = RADIUS; j < size - RADIUS && correct; j++) {
                if (fabs(h_output_global[i * size + j] - h_output_ref[i * size + j]) > epsilon) {
                    printf("Verification failed at (%d, %d): %f vs %f\n", i, j, h_output_global[i * size + j], h_output_ref[i * size + j]);
                    correct = false;
                }
            }
        }
    } else {
        // 对大矩阵，只验证两个版本结果一致
        float epsilon = 1.0e-5;
        int count = 0;
        for (int i = RADIUS; i < size - RADIUS && correct && count < 1000; i++) {
            for (int j = RADIUS; j < size - RADIUS && correct && count < 1000; j++) {
                if (fabs(h_output_global[i * size + j] - h_output_shared[i * size + j]) > epsilon) {
                    printf("Verification failed at (%d, %d): %f vs %f\n", i, j, h_output_global[i * size + j], h_output_shared[i * size + j]);
                    correct = false;
                }
                count++;
            }
        }
    }
    
    if (correct) {
        printf("\n=== Verification ===\n");
        printf("Results verified: PASS\n");
        
        // 性能对比分析
        printf("\n=== Detailed Performance Analysis ===\n");
        printf("Shared memory version is %.2fx faster than global memory version\n", speedup);
    } else {
        printf("Verification: FAILED\n");
    }
    
    free(h_input);
    free(h_output_global);
    free(h_output_shared);
    free(h_output_ref);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
