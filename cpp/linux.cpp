#include <stdio.h>
#include <unistd.h>
#include <iostream>

int main()
{
    for (int i = 0; i < 2; i++)
    {
        fork();
        // printf("fork-\n");
        std::cout << "fork\n";
    }
}

/*
cd cpp;g++ -g -std=c++17 linux.cpp -o linux;./linux;cd ..
cd cpp;rm -rf linux;cd ..
*/