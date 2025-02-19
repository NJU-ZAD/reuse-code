#include <stdio.h>
#include "head.hpp"

/* 每个SM上常量内存大小限制为64KB
1.常量内存的单次读操作可以广播到“邻近”线程，从而降低内存读操作的次数。
2.常量内存拥有高速缓存，对于相同内存地址的连续操作不会产生额外的开销。*/
float *_data;
__constant__ float constData[DATA_SIZE];
__device__ float *deviceData;

__device__ void devKernel()
{
    int threadId = threadIdx.x;
    deviceData[threadId] = threadId + constData[threadId];
    printf("constData-threadId-%d-%f\n", threadId, constData[threadId]);
    printf("deviceData-threadId-%d-%f\n", threadId, deviceData[threadId]);
}