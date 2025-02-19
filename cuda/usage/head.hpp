#pragma once
#define DATA_SIZE 32
extern float *_data;
extern __constant__ float constData[DATA_SIZE];
extern __device__ float *deviceData;
extern __device__ void devKernel();