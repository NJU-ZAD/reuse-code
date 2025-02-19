#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

int main()
{
    //定义矩阵的长度
    int Ndim = 2048, Mdim = 2048, Pdim = 2048;
    int szA = Ndim * Pdim;
    int szB = Pdim * Mdim;
    int szC = Ndim * Mdim;

    float *A;
    float *B;
    float *C;

    A = (float *)malloc(szA * sizeof(float));
    B = (float *)malloc(szB * sizeof(float));
    C = (float *)malloc(szC * sizeof(float));
    int i, j, k;
    float tmp;
    //初始化矩阵，可加学号
    for (i = 0; i < szA; i++)
        A[i] = 8;
    for (i = 0; i < szB; i++)
        B[i] = 3;

    //实现矩阵相乘
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
        {
            tmp = 0.0;
            for (k = 0; k < Pdim; k++)
                tmp += A[i * Pdim + k] * B[k * Mdim + j];
            C[i * Mdim + j] = tmp;
        }
    }

    printf("\nArray C:\n");
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
            printf("%.1f\t", C[i * Mdim + j]);
        printf("\n");
    }
    if (A)
        free(A);
    if (B)
        free(B);
    if (C)
        free(C);
    return 0;
}
/*
cd cpp;g++ -g -std=c++17 matrix_cpu.cpp -o matrix_cpu;./matrix_cpu;cd ..
cd cpp;rm -rf matrix_cpu;cd ..
*/