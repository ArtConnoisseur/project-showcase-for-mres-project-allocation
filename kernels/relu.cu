#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void printArray(float* arr, int N) {
    int counter = 0; 
    float* iter = arr;

    cout << "Calculated Size:" << sizeof(arr) << endl;

    cout << "[ ";

    while (counter < N) {
        cout << *iter << ", "; 
        iter++; 
        counter ++; 
    }

    cout << "\b\b ]\n";
}

__device__ float max_float(float a, float b) { return a > b ? a : b; }

__global__ void relu(const float* input, float* output, int size) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    output[threadId] = max_float(0, input[threadId]);
}

__global__ void leaky_relu(const float* input, float* output, int size) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    output[threadId] = input[threadId] > 0 ? input[threadId] : 0.01 * input[threadId];
}

int main() {
    const int input_size = 6;
    const float h_input[input_size] = {2.0, -9, -4.2, 0, 56.1, 2.2};
    float* h_output = (float*) malloc(input_size * sizeof(float));
    cout << "Debug size:" << sizeof(*h_output) << " " << input_size * sizeof(float) << endl;
    
    float* d_input; 
    float* d_output; 

    cudaMalloc(&d_input, sizeof(float) * input_size);
    cudaMalloc(&d_output, sizeof(float) * input_size);

    cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocksPerGrid = (input_size + threads_per_block - 1) / threads_per_block;

    leaky_relu<<<blocksPerGrid, threads_per_block>>>(d_input, d_output, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float) * input_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    printArray(h_output, input_size);
    return 0;
}