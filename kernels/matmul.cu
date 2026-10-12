#include <cuda_runtime.h>
#include <iostream>

void printMatrix(float* mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mat[i * M + j] << ", ";
        }
        std::cout << std::endl;
    }
}

__global__ void matmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, int N, int K) {

        int id_x = blockDim.x * blockIdx.x + threadIdx.x;
        int id_y = blockDim.y * blockIdx.y + threadIdx.y; 

        if (id_x < K && id_y < M) {
            float sum = 0;
            for (int i = 0; i < N; i++) sum += A[id_y * N + i] * B[i * N + id_x];
            C[id_y * K + id_x] = sum;
        }
}

int main() {
    int M = 2;
    int N = 2;
    int K = 2; 

    float *h_A, *h_B, *h_C;
    h_A = (float*) malloc(sizeof(float) * M * N);
    h_B = (float*) malloc(sizeof(float) * N * K);
    h_C = (float*) malloc(sizeof(float) * M * K);

    for (int i = 0; i < 4; i++) h_A[i] = i + 1; 
    for (int i = 0; i < 4; i++) h_B[i] = i + 5;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * M * N);
    cudaMalloc(&d_B, sizeof(float) * N * K);
    cudaMalloc(&d_C, sizeof(float) * M * K);
    
    cudaMemcpy(d_A, h_A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (M + threadsPerBlock.y - 1)/threadsPerBlock.y
    );

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printMatrix(h_A, M, N);
    printMatrix(h_B, N, K);
    printMatrix(h_C, M, K);

    cudaDeviceSynchronize();
    return 0; 
}