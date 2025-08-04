#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }
    
    std::cout << "Found " << deviceCount << " CUDA-capable device(s)." << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 cache size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Memory access alignment: " << prop.textureAlignment << " bytes" << std::endl;
        std::cout << "  Concurrent copy and execution: " << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
        
        // Memory transaction information
        std::cout << "  Memory transaction size (warp): " << prop.warpSize * sizeof(float) << " bytes" << std::endl;
        
        // The actual memory transaction size varies by GPU architecture:
        // - Older architectures (compute capability < 2.0): 32 bytes for reads, 64 bytes for writes
        // - Fermi (compute capability 2.x): 128 bytes for both reads and writes
        // - Kepler (compute capability 3.x): 128 bytes for both reads and writes
        // - Maxwell (compute capability 5.x): 128 bytes for both reads and writes
        // - Pascal (compute capability 6.x): 128 bytes for both reads and writes
        // - Volta/Turing (compute capability 7.x): 128 bytes for both reads and writes
        // - Ampere (compute capability 8.x): 128 bytes for both reads and writes
        
        std::cout << "  Approximate memory transaction size: ";
        if (prop.major < 2) {
            std::cout << "32 bytes (reads), 64 bytes (writes)" << std::endl;
        } else {
            std::cout << "128 bytes (both reads and writes)" << std::endl;
        }
    }
    
    return 0;
}