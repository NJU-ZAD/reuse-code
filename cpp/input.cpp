#include <stdio.h>
#include "../hpp/input.hpp"

int main()
{
    readCharSet(5);
    car(100);
    return 0;
}

/*
cd cpp;g++ -c -std=c++17 _input/input1.cpp -o input1.o;g++ -c -std=c++17 _input/input2.cpp -o input2.o;g++ -g -std=c++17 input.cpp -o input input1.o input2.o;./input;cd ..
cd cpp;rm -rf input1.o input2.o input;cd ..
*/