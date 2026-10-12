#include <cuda_runtime.h>
#include <iostream>

void printMatrix(float* mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mat[i * N + j] << ", ";
        }
        std::cout << std::endl;
    }
}


__global__ void matmulTKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, int N, int K) {

        int id_x = blockDim.x * blockIdx.x + threadIdx.x;
        int id_y = blockDim.y * blockIdx.y + threadIdx.y; 

        if (id_x < K && id_y < M) {
            float sum = 0;
            for (int i = 0; i < K; i++) sum += A[id_y * K + i] * B[id_x * K + i];
            C[id_y * N + id_x] =  sum / sqrtf((float) K);
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
            for (int i = 0; i < N; i++) sum += A[id_y * N + i] * B[i * K + id_x];
            C[id_y * K + id_x] = sum;
        }
}

__shared__ float smem[1024];
__shared__ float nmem[1024];

__global__ void redSoftmax(float* input, float* output, const int M, const int N) { 
    int id = threadIdx.x;
    int row = blockIdx.x;

    float norm = 0.0f; 
    float l_max = -INFINITY;

    float* input_row = input + row * N; 
    float* output_row = row * N + output;

    if (row >= M) return;

    for (int i = id; i < N; i += blockDim.x) {
        float x = input_row[i];
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
            float m1 = smem[id],          n1 = nmem[id];
            float m2 = smem[id + stride], n2 = nmem[id + stride];

            float nmax = fmaxf(m1, m2);
            float f1 = (m1 == -INFINITY) ? 0.0f : expf(m1 - nmax);
            float f2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - nmax);

            smem[id] = nmax;
            nmem[id] = n1 * f1 + n2 * f2;
        }
        __syncthreads();
    }

    float global_norm = nmem[0];
    float global_max = smem[0];
    __syncthreads();

    for (int i = id; i < N; i += blockDim.x) output_row[i] = expf(input_row[i] - global_max)/global_norm;

}

int main() {
    const int M = 2;
    const int N = 3;
    const int d = 4; // this is K in M, N, K notation of matmul

    const size_t nQ = M * d, nK = N * d, nV = N * d;
    const size_t nS = M * N;     
    const size_t nOut = M * d;

    float h_Q[M * d] = {
        1, 0, 0, 0,
        0, 1, 0, 0};
    float h_K[N * d] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0};
    float h_V[N * d] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12};
    float h_out[M * d] = {0};

    float *d_Q, *d_K, *d_V, *d_S, *d_Sout, *d_out;
    cudaMalloc(&d_Q,nQ * sizeof(float));
    cudaMalloc(&d_K,nK * sizeof(float));
    cudaMalloc(&d_V,nV * sizeof(float));
    cudaMalloc(&d_S,nS * sizeof(float));
    cudaMalloc(&d_Sout,nS * sizeof(float));
    cudaMalloc(&d_out, nOut * sizeof(float));

    cudaMemcpy(d_Q, h_Q, nQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, nK * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, nV * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 qktThreadsPerBlock(32, 32);
    dim3 qktBlocksPerGrid(
        (N + qktThreadsPerBlock.x - 1)/qktThreadsPerBlock.x,
        (M + qktThreadsPerBlock.y - 1)/qktThreadsPerBlock.y
    );
    matmulTKernel<<<qktBlocksPerGrid, qktThreadsPerBlock>>>(d_Q, d_K, d_S, M, N, d);

    int sftmThreadsPerBlock = 256;
    int sftmBlocksPerGrid = M;
    redSoftmax<<<sftmBlocksPerGrid, sftmThreadsPerBlock>>>(d_S, d_Sout, M, N);

    dim3 outThreadsPerBlock(32, 32);
    dim3 outBlocksPerGrid(
        (d + outThreadsPerBlock.x - 1)/outThreadsPerBlock.x,
        (M + outThreadsPerBlock.y - 1)/outThreadsPerBlock.y
    );
    matmulKernel<<<outBlocksPerGrid, outThreadsPerBlock>>>(d_Sout, d_V, d_out, M, N, d);

    cudaMemcpy(h_out, d_out, nOut * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(h_out, M, d);
    cudaDeviceSynchronize();

    return 0;
}