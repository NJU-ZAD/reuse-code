#include <stdio.h>
#include <cuda_runtime.h>
#include <bits/stdint-uintn.h>
#include <unistd.h>

#include "../../hpp/skills.hpp"

#define DATA_SIZE 1024
#define WARP_SIZE 32
#define THREADS_PER_GROUP (WARP_SIZE * 2)
#define GROUP_NB 2
#define TASK_NB 4

static float *data;
static __device__ float *devData;
static size_t pitch;
static __device__ size_t devPitch;
static char c;
static __device__ char devC;
static __device__ const int arrayNb = 2;

__device__ void group_task(int groupId, int _threadId)
{
	int index;
	float input = 0;
	float output = 0;
	for (int i = 0; i < DATA_SIZE; i++)
	{
		index = _threadId * devPitch / sizeof(float) + i;
		// if (_threadId == 0)
		// 	printf("groupId %d中的threadId %d正在处理第 %d 列数据 %f\n",
		// 		   groupId, _threadId, index, devData[index]);
		input = devData[index];
		output = 1;
		switch (groupId)
		{
		case 0:
			for (int j = 0; j < 1000; j++)
			{
				output *= input;
				output += input;
			}
			break;
		case 1:
			for (int j = 0; j < 2000; j++)
			{
				output *= input;
				output += input;
			}
			break;
		default:
			break;
		}
	}
}

__global__ void cond_syn()
{
	/*该线程的ID*/
	int threadId = threadIdx.x;

	/*该线程所在组的ID*/
	int groupId = threadId / THREADS_PER_GROUP;

	/*该线程的相对ID*/
	int _threadId = threadId % THREADS_PER_GROUP;

	/*一个关于共享变量的例子*/
	__shared__ char array[arrayNb];
	array[0] = '-';
	array[1] = '*';

	for (int i = 0; i < TASK_NB; i++)
	{
		for (int j = 0; j < GROUP_NB; j++)
		{
			if (j == groupId)
			{
				printf("%c%c groupId %d中的threadId %d正在执行任务%d\n",
					   array[0], array[1], groupId, threadId, i);
				group_task(groupId, _threadId);

				// 同步当前线程块中所有能够到达此处的线程
				__syncthreads();
				if (_threadId == 0)
				{
					printf("%c groupId-%d完成任务%d\n", devC, groupId, i);
				}
				break;
			}
		}
	}
}

void generate_data()
{
	float *hostData = (float *)malloc(THREADS_PER_GROUP * DATA_SIZE * sizeof(float));
	for (int i = 0; i < THREADS_PER_GROUP; i++)
	{
		for (int j = 0; j < DATA_SIZE; j++)
		{
			int index = i * DATA_SIZE + j;
			hostData[index] = random() / pow(10, 3);
			hostData[index] = index / pow(10, 3);
		}
	}
	cudaMallocPitch(&data, &pitch, DATA_SIZE * sizeof(float), THREADS_PER_GROUP);
	printf("pitch=%lu\n", pitch);
	cudaMemcpy2D(data, pitch, hostData, DATA_SIZE * sizeof(float),
				 DATA_SIZE * sizeof(float), THREADS_PER_GROUP, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(devData, &data, sizeof(data));
	c = '$';
	cudaMemcpyToSymbol(devC, &c, sizeof(c));
	cudaMemcpyToSymbol(devPitch, &pitch, sizeof(pitch));
	free(hostData);
}

int main(int argc, char *argv[])
{
	char path[256];
	getWorkDir(path, sizeof(path), true);
	changeWorkDir(argv);
	char fileName[256] = "cond_syn.log";
	remove(fileName);

	generate_data();

	int stdDup = dup(1);
	FILE *outLog = fopen(fileName, "a");
	dup2(fileno(outLog), 1);

	cond_syn<<<1, GROUP_NB * THREADS_PER_GROUP>>>();
	// /*如果不加这句话main函数将不等cond_syn执行直接结束*/
	cudaDeviceSynchronize();

	fflush(stdout);
	fclose(outLog);
	dup2(stdDup, 1);
	close(stdDup);
	cudaFree(data);
	return 0;
}
////////////////////////////////////////////////////////////////////////////////
// cd cuda/cond_syn;make run;cd ../..
// cd cuda/cond_syn;make clean;cd ../..
////////////////////////////////////////////////////////////////////////////////