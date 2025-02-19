#include <stdio.h>
#include <cuda_runtime.h>
#include "head.hpp"

__global__ void run()
{
    devKernel();
}

int main()
{
    int bytes = DATA_SIZE * sizeof(float);
    float data[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = (i + 10) * 3.689;
        printf("START-%f\n", data[i]);
    }
    cudaMemcpyToSymbol(constData, &data, sizeof(data));
    cudaMalloc((void **)&_data, bytes);
    cudaMemcpy(_data, data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceData, &_data, sizeof(_data));

    run<<<1, DATA_SIZE>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(data, constData, bytes);
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf("MID-%f\n", data[i]);
    }
    cudaMemcpy(data, _data, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf("END-%f\n", data[i]);
    }

    cudaFree(_data);
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
// cd cuda/usage;make run;cd ../..
// cd cuda/usage;make clean;cd ../..
////////////////////////////////////////////////////////////////////////////////