#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void hello_kernel()
{
    printf("Hello from GPU thread %d\n", hipThreadIdx_x);
}

int main()
{
    hipLaunchKernelGGL(hello_kernel, dim3(1), dim3(32), 0, 0);
    hipDeviceSynchronize();
    printf("Hello from host\n");
    return 0;
}
