// This implementation of softmax modifies the equation to avoid
// overflow or underflow but subtracting max(x), where x is the 
// input vector 

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

void printArray(float* arr, int N) {
    for (int i = 0; i < N; i++) std::cout << arr[i] << ", ";
    std::cout << "\b\b\n";
}

__global__ void softmax(const float* input, float* output, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id < N) {
        float max_i = -INFINITY;
        float norm = 0.0;

        for (int i = 0; i < N; i ++) {
            if (input[i] > max_i) {
                norm = norm * expf(max_i - input[i]); 
                max_i = input[i];
            }

            norm += expf(input[i] - max_i);
        }
        output[id] = expf(input[id] - max_i) / norm;
    }
}

__shared__ float smem[1024];
__shared__ float nmem[1024];

__global__ void redSoftmax(const float* input, float* output, const int N) { 
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N) {
        float norm = 0.0f; 
        float l_max = -INFINITY;

        for (int i = id; i < N; i += blockDim.x) {
            float x = input[i];
            if (x > l_max) {
                norm *= expf(l_max - x);
                l_max = x;
            }
            norm += expf(x - l_max);
        }
        __syncthreads(); 

        smem[id] = l_max;
        nmem[id] = norm;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (id < stride) {
                float nmax = max(smem[id], smem[id + stride]);
                nmem[id] *= expf(smem[id] - nmax);
                nmem[id + stride] *= expf(smem[id] - nmax);
                nmem[id] = nmem[id] + nmem[id + stride];
                smem[id] = nmax;
            }
            __syncthreads(); 
        }

        float global_norm = nmem[0];
        float global_max = smem[0];
        __syncthreads();

        output[id] = expf(input[id] - global_max)/global_norm;
    }
}

int main() {
    const int threadsPerBlock = 256;
    const int N = 5;
    float* h_input = (float*) malloc(N * sizeof(float));
    float* h_output = (float*) malloc(N * sizeof(float));

    for ( int i = 0; i < N; i++ ) { h_input[i] = 5 * (i - 2); }

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    const int gridSize = (N + threadsPerBlock - 1) / threadsPerBlock;
    redSoftmax<<<gridSize, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_input);
    
    cudaDeviceSynchronize();

    printArray(h_output, N);
    free(h_input);
    free(h_output);
    
    return 0;
}