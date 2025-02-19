#include <stdio.h>
#include <cuda_runtime.h>
#include <bits/stdint-uintn.h>
#include <unistd.h>

#include "../../hpp/skills.hpp"

#define N 9999
static int NUM_OF_SM, NUM_OF_THREAD;
#define nStreams 5
static cudaStream_t stream[nStreams];
#define Test 3
static cudaEvent_t startEvent[Test], stopEvent[Test];
static float runTime[Test];
static double *hostData, *devData;

__device__ __inline__ uint32_t getSmid()
{
	uint32_t smId;
	asm volatile("mov.u32 %0, %%smid;"
				 : "=r"(smId));
	return smId;
}

__global__ void runTask(int smId, double *devData)
{
	uint32_t currSmId = getSmid();
	uint32_t blockSize = blockDim.x;
	uint32_t blockId = blockIdx.x;
	uint32_t threadId = threadIdx.x;

	if (smId == -1 || smId == currSmId)
	{
		int j = N / blockSize;
		for (int i = 0; i < j; i++)
		{
			for (int k = 0; k < N; k++)
			{
				devData[i] = (devData[i] + threadId) / (blockId + 1);
			}
		}
	}
}

void test(int I)
{
	if (I == 0)
	{
		printf("%d个线程块（线程数量为%d）在相同CUDA流中执行\n", nStreams, NUM_OF_THREAD);
		for (int i = 0; i < nStreams; i++)
		{
			runTask<<<1, NUM_OF_THREAD>>>(-1, devData);
		}
	}
	else if (I == 1)
	{
		int smId = 3;
		printf("%d个线程块（线程数量为%d）在不同CUDA流中执行（强制绑定SM %d）\n", nStreams, NUM_OF_THREAD, smId);
		for (int i = 0; i < nStreams; i++)
		{
			runTask<<<NUM_OF_SM, NUM_OF_THREAD, 0, stream[i]>>>(smId, devData);
		}
	}
	else if (I == 2)
	{
		printf("%d个线程块（线程数量为%d）在不同CUDA流中执行\n", nStreams, NUM_OF_THREAD);
		for (int i = 0; i < nStreams; i++)
		{
			runTask<<<1, NUM_OF_THREAD, 0, stream[i]>>>(-1, devData);
		}
	}
}

int main(int argc, char *argv[])
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	NUM_OF_SM = deviceProp.multiProcessorCount;
	NUM_OF_THREAD = deviceProp.maxThreadsPerBlock;
	printf("NUM_OF_SM %d\n", NUM_OF_SM);
	printf("NUM_OF_THREAD %d\n", NUM_OF_THREAD);

	for (int i = 0; i < nStreams; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
	for (int i = 0; i < Test; i++)
	{
		cudaEventCreate(&startEvent[i]);
		cudaEventCreate(&stopEvent[i]);
	}

	int byteSize = sizeof(double) * N;
	hostData = (double *)malloc(byteSize);
	memset(hostData, 0, byteSize);
	cudaMalloc((void **)&devData, byteSize);
	cudaMemcpy(devData, hostData, byteSize, cudaMemcpyHostToDevice);

	for (int i = 0; i < Test; i++)
	{
		cudaEventRecord(startEvent[i], 0);
		test(i);
		cudaEventRecord(stopEvent[i], 0);
		cudaEventSynchronize(stopEvent[i]);
		cudaEventElapsedTime(&runTime[i], startEvent[i], stopEvent[i]);
		printf("总时间：%fms\n", runTime[i]);
	}

	cudaMemcpy(hostData, devData, byteSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < nStreams; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
	for (int i = 0; i < Test; i++)
	{
		cudaEventDestroy(startEvent[i]);
		cudaEventDestroy(stopEvent[i]);
	}
	free(hostData);
	cudaFree(devData);

	return 0;
}
////////////////////////////////////////////////////////////////////////////////
// cd cuda/concurrency;make run;cd ../..
// cd cuda/concurrency;make clean;cd ../..
////////////////////////////////////////////////////////////////////////////////