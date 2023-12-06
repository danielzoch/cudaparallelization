#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int N = 1024;  // Adjust based on your system capabilities

__global__ void vectorAddWithSharedMemory(int *a, int *b, int *result) {
    __shared__ int sharedResult[N];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    sharedResult[threadIdx.x] = (tid < N) ? a[tid] + b[tid] : 0;

    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < N) {
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = sharedResult[0];
    }
}

__global__ void vectorAddWithoutSharedMemory(int *a, int *b, int *result) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    result[tid] = (tid < N) ? a[tid] + b[tid] : 0;
}

int main() {
    int *h_a, *h_b, *h_result_with_shared, *h_result_without_shared;

    h_a = new int[N];
    h_b = new int[N];
    h_result_with_shared = new int[1];  // Result for shared memory is a single value
    h_result_without_shared = new int[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    int *d_a, *d_b, *d_result_with_shared, *d_result_without_shared;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_result_with_shared, sizeof(int));  // Allocate space for a single result
    cudaMalloc((void**)&d_result_without_shared, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(N);
    dim3 gridDim(1);

    // Timing for sum with shared memory
    auto start_time_with_shared = std::chrono::high_resolution_clock::now();
    vectorAddWithSharedMemory<<<gridDim, blockDim>>>(d_a, d_b, d_result_with_shared);
    cudaDeviceSynchronize();
    auto end_time_with_shared = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_with_shared = end_time_with_shared - start_time_with_shared;
    std::cout << "Time with shared memory: " << elapsed_time_with_shared.count() << " seconds\n";

    // Timing for sum without shared memory
    auto start_time_without_shared = std::chrono::high_resolution_clock::now();
    vectorAddWithoutSharedMemory<<<gridDim, blockDim>>>(d_a, d_b, d_result_without_shared);
    cudaDeviceSynchronize();
    auto end_time_without_shared = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time_without_shared = end_time_without_shared - start_time_without_shared;
    std::cout << "Time without shared memory: " << elapsed_time_without_shared.count() << " seconds\n";

    // Retrieve results
    cudaMemcpy(h_result_with_shared, d_result_with_shared, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_without_shared, d_result_without_shared, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate the results
    int sum_with_shared = h_result_with_shared[0];
    int sum_without_shared = 0;
    for (int i = 0; i < N; ++i) {
        sum_without_shared += h_result_without_shared[i];
    }

    std::cout << "Sum with shared memory: " << sum_with_shared << std::endl;
    std::cout << "Sum without shared memory: " << sum_without_shared << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_result_with_shared;
    delete[] h_result_without_shared;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result_with_shared);
    cudaFree(d_result_without_shared);

    return 0;
}

