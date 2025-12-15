#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main(int argc, char** argv)
{
    const int N = argc > 1 ? atoi(argv[1]) : 2048;
    const int iterations = argc > 2 ? atoi(argv[2]) : 2000;
    const bool quiet = getenv("HIP_GEMM_QUIET") != nullptr;

    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    float* hA = reinterpret_cast<float*>(malloc(bytes));
    float* hB = reinterpret_cast<float*>(malloc(bytes));
    float* hC = reinterpret_cast<float*>(malloc(bytes));

    for (int i = 0; i < N * N; ++i) {
        hA[i] = sinf(i);
        hB[i] = cosf(i);
        hC[i] = 0.0f;
    }

    float *dA, *dB, *dC;
    hipMalloc(&dA, bytes);
    hipMalloc(&dB, bytes);
    hipMalloc(&dC, bytes);

    hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    for (int iter = 0; iter < iterations; ++iter) {
        hipLaunchKernelGGL(matmul_kernel, grid, block, 0, 0, dA, dB, dC, N);
        hipDeviceSynchronize();
    }

    hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float expected = 0;
            for (int k = 0; k < N; ++k) {
                expected += hA[row * N + k] * hB[k * N + col];
            }
            max_diff = fmaxf(max_diff, fabsf(hC[row * N + col] - expected));
        }
    }

    if (!quiet) {
        printf("max error sample = %.6f\n", max_diff);
    }

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    free(hA);
    free(hB);
    free(hC);

    if (max_diff > 1e-3f) {
        fprintf(stderr, "result mismatch\n");
        return 1;
    }

    if (!quiet) {
        printf("Matrix multiply PFLOP stress passed\n");
    }
    return 0;
}
