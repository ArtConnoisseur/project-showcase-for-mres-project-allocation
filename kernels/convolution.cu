// 1D Convolution
// Easy

// Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.
// input 1 2 3 4 5 kernel 1 0 -1 1×1 2×0 3×(-1) = 1 + 0 + (-3) = -2 output -2 …

// The input consists of two arrays:

//     input: A 1D array of 32-bit floating-point numbers.
//     kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.

// The output should be written to the output array, which will have a size of input_size - kernel_size + 1.

// The convolution operation is defined mathematically as:

// where ranges from 0 to .
// Implementation Requirements

//     Use only native features (external libraries are not permitted)
//     The solve function signature must remain unchanged
//     The final result must be stored in the array output

// Example 1:

// Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
// Output: [-2, -2, -2]

// Example 2:

// Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
// Output: [1.8, 3.2, 4.6]

// Constraints

//     1 ≤ input_size ≤ 1,500,000
//     1 ≤ kernel_size ≤ 2047
//     kernel_size ≤ input_size
//     Performance is measured with input_size = 1,500,000, kernel_size = 2,047

#include <cuda_runtime.h>
#include <stdio.h>

void printArray(float* arr, int arrSize) {
    printf("[ ");
    for (int i = 0; i < arrSize; i++) printf("%f, ", arr[i]);
    printf("\b\b ]\n");
}

__global__ void convolutionKernel(
    const float* input,
    const float* kernel,
    float* output,
    int inputSize,
    int kernelSize
) {
    const int id = (blockDim.x * blockIdx.x + threadIdx.x);
    if (id < inputSize + kernelSize + 1) {
        float sum = 0;
        for (int i = 0; i < kernelSize; i++) sum += input[id + i] * kernel[i];
        output[id] = sum;
    }
}

int main() {
    // Base stuff we need for the problem 
    const int inputSize = 5;
    const int kernelSize = 3;
    const int outputSize = inputSize - kernelSize + 1; 

    const int inputBytes = inputSize * sizeof(float);
    const int kernelBytes = kernelSize * sizeof(float);
    const int outputBytes = outputSize * sizeof(float);

    // Defining CPU mem
    float *input = (float*) malloc(inputBytes);
    float *kernel = (float*) malloc(kernelBytes);
    
    for (int i = 0; i < inputSize; i++) input[i] = i;
    
    kernel[0] = 1.0;
    kernel[1] = 0.0;
    kernel[2] = -1.0;
   
    
    float *output = (float*) malloc(outputBytes);
    
    // Defining device mem
    float *devi, *devk, *devo; 
    cudaMalloc(&devi, inputBytes);
    cudaMalloc(&devk, kernelBytes);
    cudaMalloc(&devo, outputBytes);
    
    // Copying from host to device 
    cudaMemcpy(devi, input, inputBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devk, kernel, kernelBytes, cudaMemcpyHostToDevice);
    
    // Kernel Primitives 
    const int threadsPerBlock = 256;
    
    // This should be output size because the operation needs to 
    // happen outputSize times. 
    const int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        devi,
        devk,
        devo,
        inputSize,
        kernelSize
    );
    
    
    // Copy result back to host
    cudaMemcpy(output, devo, outputBytes, cudaMemcpyDeviceToHost);
    printArray(output, outputSize);
    
    // Free memory
    cudaFree(devi);
    cudaFree(devo);
    cudaFree(devo);

    cudaDeviceSynchronize();
    return 0; 
}