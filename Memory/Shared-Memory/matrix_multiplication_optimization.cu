#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

void initMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i * size + j] = (float)rand() / RAND_MAX;
        }
    }
}

// 原始的全局内存版本矩阵乘法
__global__ void matrixMulGlobal(float* C, float* A, float* B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// 优化的共享内存版本矩阵乘法
__global__ void matrixMulShared(float* C, float* A, float* B, int width) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < (width + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // 边界检查，防止访问越界
        if (row < width && (m * TILE_SIZE + tx) < width)
            sA[ty][tx] = A[row * width + (m * TILE_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;
            
        if ((m * TILE_SIZE + ty) < width && col < width)
            sB[ty][tx] = B[(m * TILE_SIZE + ty) * width + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
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
    const int width = 1024;
    size_t size = width * width * sizeof(float);
    
    printf("Matrix size: %d x %d\n", width, width);
    printf("Data size per matrix: %.2f MB\n", (float)size / (1024.0f * 1024.0f));
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);
    
    initMatrix(h_A, width);
    initMatrix(h_B, width);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    
    // 测试全局内存版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 多次运行取平均值以提高测量准确性
    const int numRuns = 5;
    float totalTimeGlobal = 0.0f;
    
    for (int run = 0; run < numRuns; run++) {
        cudaEventRecord(start);
        matrixMulGlobal<<<grid, block>>>(d_C, d_A, d_B, width);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTimeGlobal += milliseconds;
    }
    
    float avgTimeGlobal = totalTimeGlobal / numRuns;
    printf("\n=== Performance Comparison ===\n");
    printf("Global memory version: %.3f ms (average of %d runs)\n", avgTimeGlobal, numRuns);
    
    cudaMemcpy(h_C_ref, d_C, size, cudaMemcpyDeviceToHost);
    
    // 测试共享内存版本
    float totalTimeShared = 0.0f;
    
    for (int run = 0; run < numRuns; run++) {
        cudaEventRecord(start);
        matrixMulShared<<<grid, block>>>(d_C, d_A, d_B, width);
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
    // 对于矩阵乘法，我们读取两个矩阵并写入一个矩阵
    size_t totalDataTransferred = 3 * size;
    float bandwidthGlobal = calculateBandwidth(totalDataTransferred, avgTimeGlobal);
    float bandwidthShared = calculateBandwidth(totalDataTransferred, avgTimeShared);
    
    printf("Effective bandwidth (global memory): %.2f GB/s\n", bandwidthGlobal);
    printf("Effective bandwidth (shared memory): %.2f GB/s\n", bandwidthShared);
    printf("Bandwidth improvement: %.2fx\n", bandwidthShared / bandwidthGlobal);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 验证结果正确性
    bool correct = true;
    float epsilon = 1.0e-3; // 放宽容差以适应浮点运算差异
    for (int i = 0; i < width && correct; i++) {
        for (int j = 0; j < width && correct; j++) {
            if (fabs(h_C[i * width + j] - h_C_ref[i * width + j]) > epsilon) {
                printf("Verification failed at (%d, %d): %f vs %f\n", i, j, h_C[i * width + j], h_C_ref[i * width + j]);
                correct = false;
            }
        }
    }
    
    if (correct) {
        printf("\n=== Verification ===\n");
        printf("Results verified: PASS\n");
        
        // 性能对比分析
        printf("\n=== Detailed Performance Analysis ===\n");
        printf("Shared memory version is %.2fx faster than global memory version\n", speedup);
        
        // 计算加速比分析
        if (speedup > 2.0) {
            printf("This significant speedup (>2x) clearly demonstrates the effectiveness of data reuse through shared memory.\n");
            printf("By storing frequently accessed data in shared memory, we reduce global memory accesses,\n");
            printf("which have higher latency compared to shared memory accesses.\n");
        } else if (speedup > 1.5) {
            printf("This moderate speedup (>1.5x) shows good benefit from shared memory data reuse.\n");
            printf("The performance gain comes from reducing global memory traffic through data reuse.\n");
        } else if (speedup > 1.0) {
            printf("This modest speedup shows some benefit from shared memory data reuse.\n");
            printf("The performance gain may be limited by other factors such as memory bandwidth or computation intensity.\n");
        } else {
            printf("There is little to no speedup, which may indicate that the workload is not memory-bound\n");
            printf("or that there are other bottlenecks in the implementation.\n");
        }
        
   } else {
        printf("Verification: FAILED\n");
    }
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
