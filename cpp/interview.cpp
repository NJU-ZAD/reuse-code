#include <iostream>
using namespace std;

void question1()
{
    // 大部分数都出现偶数次，只有两个数出现奇数次
    // a⊕a=0
    // a⊕0=a
    int arr[200], nb = 0, temp;
    // 回车之后Ctrl+D
    while (cin >> temp)
    {
        arr[nb++] = temp;
    }
    int res = 0, small = 0, big = 0;
    for (int i = 0; i < nb; i++)
    {
        res ^= arr[i];
    }
    int bits = 0, index = 0;
    while (res != 1)
    {
        res >>= 1;
        bits++;
    }
    index = 1 << bits;
    for (int i = 0; i < nb; i++)
    {
        if ((arr[i] & index) == 0)
            small ^= arr[i];
        else if ((arr[i] & index) == index)
            big ^= arr[i];
    }
    cout << "small=" << small << endl
         << "big=" << big << endl;
}

int main()
{
    question1();
}

/*
cd cpp;g++ -g -std=c++17 interview.cpp -o interview;./interview;cd ..
cd cpp;rm -rf interview;cd ..
*/