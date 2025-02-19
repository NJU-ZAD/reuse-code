#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define index 2048
#define TILE_WIDTH 2

__global__ void calcSum(float *AA, float *BB, float *CC, int Width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < Width) && (Col < Width))
    {
        float Pvalue = 0;
        // 每一个线程处理输出数据d_P的一个元素
        for (int k = 0; k < Width; ++k)
            Pvalue += AA[Row * Width + k] * BB[k * Width + Col];
        CC[Row * Width + Col] = Pvalue;
    }
}

int main()
{
    cudaError_t cudaStatus = cudaSuccess;
    //初始化cpu矩阵
    int Ndim = 0, Mdim = 0, Pdim = 0, Width = 0;
    Ndim = Mdim = Pdim = Width = index;
    int szA = Ndim * Pdim;
    int szB = Pdim * Mdim;
    int szC = Ndim * Mdim;
    float *A, *AA;
    float *B, *BB;
    float *C, *CC;
    A = (float *)malloc(szA * sizeof(float));
    B = (float *)malloc(szB * sizeof(float));
    C = (float *)malloc(szC * sizeof(float));
    int i, j;
    for (i = 0; i < szA; i++)
        A[i] = 8;
    for (i = 0; i < szB; i++)
        B[i] = 3;

    cudaStatus = cudaMalloc((void **)&AA, szA * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc1 failed!");
    }
    cudaStatus = cudaMalloc((void **)&BB, szB * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc2 failed!");
    }
    cudaStatus = cudaMalloc((void **)&CC, szC * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc3 failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(AA, A, szA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy1 failed!");
    }
    cudaStatus = cudaMemcpy(BB, B, szB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy2 failed!");
    }

    // 线程配置
    // TILE_WIDTH 是一个用“#define”定义的常量
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    calcSum<<<dimGrid, dimBlock>>>(AA, BB, CC, Width);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "calcSum failed!");
        return 1;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, CC, szC * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    //打印
    printf("\nArray C:\n");
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
            printf("%.1f\t", C[i * Mdim + j]);
        printf("\n");
    }
    cudaFree(AA);
    cudaFree(BB);
    cudaFree(CC);
    free(A);
    free(B);
    free(C);

    return 0;
}
////////////////////////////////////////////////////////////////////////////////
// cd cuda/matrix_gpu;make run;cd ../..
// cd cuda/matrix_gpu;make clean;cd ../..
////////////////////////////////////////////////////////////////////////////////