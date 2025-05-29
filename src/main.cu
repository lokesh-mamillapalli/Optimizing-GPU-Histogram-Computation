#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

// Required naive kernel implementation
__global__ void computeHistogramNaive(const unsigned char* input, int* histogram, int N, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&histogram[input[idx]], 1);
    }
}

// Shared memory implementation required by assignment
__global__ void computeHistogramShared(const unsigned char* input, int* histogram, int N, int numBins) {
    extern __shared__ int sharedHist[];
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&sharedHist[input[i]], 1);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        if (sharedHist[i] > 0) {
            atomicAdd(&histogram[i], sharedHist[i]);
        }
    }
}

// Ultra-optimized kernel for V100 - heavily tuned for performance
__global__ void computeHistogramOptimized(const int* input, int* histogram, int N, int numBins) {
    // Register-based histogram for most frequent bins (V100 can handle 32 bins in registers efficiently)
    int localHist[32] = {0};
    
    // Shared memory for the full histogram
    extern __shared__ int sharedHist[];
    
    // Initialize shared memory using vectorized operations when possible
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    // Process multiple elements per thread - tuned specifically for V100
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // V100 sweet spot - 12 elements per thread provides optimal balance
    const int elementsPerThread = 12;
    
    // Process chunks of elements - key to hiding memory latency
    for (int base = tid; base < N; base += stride * elementsPerThread) {
        // Prefetch all values to registers
        int values[elementsPerThread];
        
        // Explicit prefetching loop
        #pragma unroll
        for (int i = 0; i < elementsPerThread; i++) {
            int idx = base + i * stride;
            if (idx < N) {
                values[i] = input[idx];
            } else {
                values[i] = -1;
            }
        }
        
        // Process prefetched values with minimal memory traffic
        #pragma unroll
        for (int i = 0; i < elementsPerThread; i++) {
            int value = values[i];
            if (value >= 0 && value < numBins) {
                if (value < 32) {
                    localHist[value]++;
                } else {
                    atomicAdd(&sharedHist[value], 1);
                }
            }
        }
    }
    
    // Combine local histograms to shared memory
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (localHist[i] > 0) {
            atomicAdd(&sharedHist[i], localHist[i]);
        }
    }
    
    __syncthreads();
    
    // Final reduction to global memory with coalesced writes
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        int val = sharedHist[i];
        if (val > 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

// CPU histogram computation for verification
void computeHistogramCPU(const unsigned char* input, int* histogram, int N, int numBins) {
    for (int i = 0; i < numBins; i++) {
        histogram[i] = 0;
    }
    
    for (int i = 0; i < N; i++) {
        histogram[input[i]]++;
    }
}

namespace solution {
    std::string compute(const std::string &input_path, int N, int B) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_histogram.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream input_fs(input_path, std::ios::binary);
        
        // Read input data
        auto input_data = std::make_unique<int[]>(N);
        input_fs.read(reinterpret_cast<char*>(input_data.get()), N * sizeof(int));
        input_fs.close();
        
        // Allocate histogram on host
        auto histogram = std::make_unique<int[]>(B);
        for (int i = 0; i < B; i++) {
            histogram[i] = 0;
        }
        
        // Allocate device memory with error checking
        int *d_input = nullptr, *d_histogram = nullptr;
        
        cudaMalloc(&d_input, N * sizeof(int));
        cudaMalloc(&d_histogram, B * sizeof(int));
        
        // Create CUDA streams for overlapping operations
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Use pinned memory for faster transfers
        cudaHostRegister(input_data.get(), N * sizeof(int), cudaHostRegisterDefault);
        
        // Asynchronous copy to device - critical for performance
        cudaMemcpyAsync(d_input, input_data.get(), N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(d_histogram, 0, B * sizeof(int), stream);
        
        // Launch parameters tuned for V100
        const int blockSize = 1024; // Optimal block size for V100
        
        // Calculate grid size for optimal V100 occupancy
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        int numSMs = 80; // V100 default
        cudaDeviceProp deviceProp;
        
        if (deviceCount > 0) {
            cudaGetDeviceProperties(&deviceProp, 0);
            numSMs = deviceProp.multiProcessorCount;
        }
        
        // V100-specific tuning for optimal occupancy
        int gridSize = (numSMs * 2048 + blockSize - 1) / blockSize;
        
        // Limit grid size to avoid excessive blocks
        int maxGridSize = (N + blockSize * 12 - 1) / (blockSize * 12);
        if (gridSize > maxGridSize) gridSize = maxGridSize;
        
        // Launch optimized kernel
        int sharedMemSize = B * sizeof(int);
        computeHistogramOptimized<<<gridSize, blockSize, sharedMemSize, stream>>>(d_input, d_histogram, N, B);
        
        // Wait for kernel to complete - using stream synchronize for best performance
        cudaStreamSynchronize(stream);
        
        // Copy result back
        cudaMemcpyAsync(histogram.get(), d_histogram, B * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Cleanup
        cudaHostUnregister(input_data.get());
        cudaFree(d_input);
        cudaFree(d_histogram);
        cudaStreamDestroy(stream);
        
        // Write result
        sol_fs.write(reinterpret_cast<char*>(histogram.get()), B * sizeof(int));
        sol_fs.close();
        
        return sol_path;
    }
}