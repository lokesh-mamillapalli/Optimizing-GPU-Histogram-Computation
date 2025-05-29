cmake_minimum_required(VERSION 3.10)
project(histogramLib CUDA CXX)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add CUDA compilation flags optimized for V100
set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS} 
    -O3 
    --use_fast_math
    -arch=sm_70  # V100 architecture
    --maxrregcount=96  # Tuned register limit for V100
    --ptxas-options=-v
    -Xptxas -O3,-dlcm=ca  # Cache all loads
    -Xcompiler -O3,-march=native,-ffast-math,-funroll-loops
)

# Create the library using cuda_add_library
cuda_add_library(studentlib STATIC main.cu)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)