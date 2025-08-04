#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

#define DATA_SIZE 1024 * 1024 * 128  // 128M elements
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

// Kernel for copying data with perfectly coalesced access (aligned)
__global__ void copyKernelCoalesced(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Kernel for copying data with strided access (2x)
__global__ void copyKernelStrided2x(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 < n) {
        output[idx * 2] = input[idx * 2];  // Strided access
    }
}

// Kernel for copying data with strided access (4x)
__global__ void copyKernelStrided4x(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 < n) {
        output[idx * 4] = input[idx * 4];  // Strided access
    }
}

// Kernel for copying data with misaligned access (+1)
__global__ void copyKernelMisaligned1(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        output[idx] = input[idx + 1];  // Misaligned access
    }
}

// Kernel for copying data with misaligned access (+4 bytes)
__global__ void copyKernelMisaligned4(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        output[idx] = input[idx + 1];  // Misaligned access (1 float = 4 bytes)
    }
}

// Kernel for copying data with misaligned access (+16 bytes)
__global__ void copyKernelMisaligned16(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 4) {
        output[idx] = input[idx + 4];  // Misaligned access (4 floats = 16 bytes)
    }
}

// Kernel for copying data with misaligned access (+32 bytes)
__global__ void copyKernelMisaligned32(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 8) {
        output[idx] = input[idx + 8];  // Misaligned access (8 floats = 32 bytes)
    }
}

// Kernel for copying data with misaligned access (+128 bytes)
__global__ void copyKernelMisaligned128(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 32) {
        output[idx] = input[idx + 32];  // Misaligned access (32 floats = 128 bytes)
    }
}

// Kernel for copying data with misaligned access (+256 bytes)
__global__ void copyKernelMisaligned256(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 64) {
        output[idx] = input[idx + 64];  // Misaligned access (64 floats = 256 bytes)
    }
}

// Kernel for copying data with misaligned access (+512 bytes)
__global__ void copyKernelMisaligned512(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 128) {
        output[idx] = input[idx + 128];  // Misaligned access (128 floats = 512 bytes)
    }
}

// Kernel for copying data with boundary-crossing access
__global__ void copyKernelBoundaryCrossing(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Access pattern that crosses memory transaction boundaries
    if (idx < n - 32) {
        output[idx] = input[(idx / 32) * 32 + ((idx % 32) + 1) % 32];
    }
}

// Kernel for copying data with random access pattern
__global__ void copyKernelRandom(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Pseudo-random access pattern
        int random_idx = (idx * 17 + 23) % (n - 1);
        output[idx] = input[random_idx];
    }
}

void runBenchmark(const std::string& testName, 
                  float* d_input, 
                  float* d_output,
                  void (*kernel)(float*, float*, int),
                  long long& avg_time,
                  int iterations = 5) {
    // Warmup
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, DATA_SIZE);
    cudaDeviceSynchronize();
    
    // Actual timing
    std::vector<long long> times(iterations);
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, DATA_SIZE);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    // Calculate average time (excluding warmup)
    avg_time = std::accumulate(times.begin(), times.end(), 0LL) / iterations;
    std::cout << testName << " average time: " << avg_time << " microseconds" << std::endl;
}

int main() {
    size_t size = DATA_SIZE * sizeof(float);
    size_t dataSizeBytes = DATA_SIZE * sizeof(float);
    
    // Host memory allocation
    float* h_input = new float[DATA_SIZE];
    
    // Initialize input data
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Run benchmarks
    std::cout << "Running benchmarks with " << DATA_SIZE << " elements (" 
              << dataSizeBytes / (1024.0 * 1024.0) << " MB)..." << std::endl;
    std::cout << "========================================================" << std::endl;
    
    long long coalesced_time, strided2x_time, strided4x_time, 
              misaligned1_time, misaligned4_time, misaligned16_time, 
              misaligned32_time, misaligned128_time, misaligned256_time, 
              misaligned512_time, boundary_time, random_time;
    
    runBenchmark("Coalesced (optimal) access", d_input, d_output, copyKernelCoalesced, coalesced_time);
    runBenchmark("Strided access (2x)", d_input, d_output, copyKernelStrided2x, strided2x_time);
    runBenchmark("Strided access (4x)", d_input, d_output, copyKernelStrided4x, strided4x_time);
    runBenchmark("Misaligned access (+1 float)", d_input, d_output, copyKernelMisaligned1, misaligned1_time);
    runBenchmark("Misaligned access (+4 bytes)", d_input, d_output, copyKernelMisaligned4, misaligned4_time);
    runBenchmark("Misaligned access (+16 bytes)", d_input, d_output, copyKernelMisaligned16, misaligned16_time);
    runBenchmark("Misaligned access (+32 bytes)", d_input, d_output, copyKernelMisaligned32, misaligned32_time);
    runBenchmark("Misaligned access (+128 bytes)", d_input, d_output, copyKernelMisaligned128, misaligned128_time);
    runBenchmark("Misaligned access (+256 bytes)", d_input, d_output, copyKernelMisaligned256, misaligned256_time);
    runBenchmark("Misaligned access (+512 bytes)", d_input, d_output, copyKernelMisaligned512, misaligned512_time);
    runBenchmark("Boundary crossing access", d_input, d_output, copyKernelBoundaryCrossing, boundary_time);
    runBenchmark("Random access", d_input, d_output, copyKernelRandom, random_time);
    
    // Calculate bandwidth
    double coalesced_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (coalesced_time / 1000000.0);
    double strided2x_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (strided2x_time / 1000000.0);
    double strided4x_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (strided4x_time / 1000000.0);
    double misaligned1_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned1_time / 1000000.0);
    double misaligned4_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned4_time / 1000000.0);
    double misaligned16_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned16_time / 1000000.0);
    double misaligned32_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned32_time / 1000000.0);
    double misaligned128_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned128_time / 1000000.0);
    double misaligned256_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned256_time / 1000000.0);
    double misaligned512_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (misaligned512_time / 1000000.0);
    double boundary_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (boundary_time / 1000000.0);
    double random_bandwidth = (dataSizeBytes / (1024.0 * 1024.0 * 1024.0)) / (random_time / 1000000.0);
    
    std::cout << "\nEffective Memory Bandwidth:" << std::endl;
    std::cout << "Coalesced access: " << coalesced_bandwidth << " GB/s" << std::endl;
    std::cout << "Strided access (2x): " << strided2x_bandwidth << " GB/s" << std::endl;
    std::cout << "Strided access (4x): " << strided4x_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+1 float): " << misaligned1_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+4 bytes): " << misaligned4_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+16 bytes): " << misaligned16_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+32 bytes): " << misaligned32_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+128 bytes): " << misaligned128_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+256 bytes): " << misaligned256_bandwidth << " GB/s" << std::endl;
    std::cout << "Misaligned access (+512 bytes): " << misaligned512_bandwidth << " GB/s" << std::endl;
    std::cout << "Boundary crossing access: " << boundary_bandwidth << " GB/s" << std::endl;
    std::cout << "Random access: " << random_bandwidth << " GB/s" << std::endl;
    
    // Calculate performance improvement
    std::cout << "\nPerformance Comparison (relative to coalesced):" << std::endl;
    std::cout << "Strided access (2x): " << coalesced_time / (double)strided2x_time << "x slower" << std::endl;
    std::cout << "Strided access (4x): " << coalesced_time / (double)strided4x_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+1 float): " << coalesced_time / (double)misaligned1_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+4 bytes): " << coalesced_time / (double)misaligned4_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+16 bytes): " << coalesced_time / (double)misaligned16_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+32 bytes): " << coalesced_time / (double)misaligned32_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+128 bytes): " << coalesced_time / (double)misaligned128_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+256 bytes): " << coalesced_time / (double)misaligned256_time << "x slower" << std::endl;
    std::cout << "Misaligned access (+512 bytes): " << coalesced_time / (double)misaligned512_time << "x slower" << std::endl;
    std::cout << "Boundary crossing access: " << coalesced_time / (double)boundary_time << "x slower" << std::endl;
    std::cout << "Random access: " << coalesced_time / (double)random_time << "x slower" << std::endl;
    
    // Calculate bandwidth improvement
    std::cout << "\nBandwidth Comparison (relative to coalesced):" << std::endl;
    std::cout << "Strided access (2x): " << (strided2x_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Strided access (4x): " << (strided4x_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+1 float): " << (misaligned1_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+4 bytes): " << (misaligned4_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+16 bytes): " << (misaligned16_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+32 bytes): " << (misaligned32_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+128 bytes): " << (misaligned128_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+256 bytes): " << (misaligned256_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Misaligned access (+512 bytes): " << (misaligned512_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Boundary crossing access: " << (boundary_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    std::cout << "Random access: " << (random_bandwidth / coalesced_bandwidth) * 100 << "%" << std::endl;
    
    // Explanation of results
    std::cout << "\nExplanation:" << std::endl;
    std::cout << "1. Coalesced access: Each thread accesses consecutive memory locations." << std::endl;
    std::cout << "   This is the most efficient pattern for GPU memory access." << std::endl;
    std::cout << "2. Strided access: Threads access elements with fixed intervals, reducing memory efficiency." << std::endl;
    std::cout << "   Larger strides result in lower efficiency." << std::endl;
    std::cout << "3. Misaligned access: May be optimized by hardware or cache." << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return 0;
}
