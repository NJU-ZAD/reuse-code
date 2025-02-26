#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <bits/stdint-uintn.h>
#include <typeinfo>
#include <assert.h>
#include <time.h>

#include "../hpp/skills.hpp"

#define __zad__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define PRINTF_TO_FILE
enum move
{
    LEFT,
    RIGHT,
    UP,
    DOWN
};

void newFolder(char *dirName)
{
    struct stat st = {0};
    if (stat(dirName, &st) == -1)
    {
        mkdir(dirName, 0700);
    }
}

void deleteFolder(char *dirName)
{
    struct stat st = {0};
    if (stat(dirName, &st) != -1)
    {
        rmdir(dirName);
    }
}

void deleteFile(char *fileName)
{
    FILE *file;
    if (file = fopen(fileName, "r"))
    {
        remove(fileName);
    }
}

void strCat(int number)
{
    char _str[8];
    sprintf(_str, "%02d", number);
    char str[15] = "name";
    int len = sizeof("name");
    mempcpy(_str + 2, ".txt", sizeof(".txt"));
    mempcpy(str + len - 1, _str, sizeof(_str));
    printf("%s\n", str);
}

void redirectPrintf()
{
    printf("[终端] 所有printf的输出信息输出到终端\n");

    char fileName[15] = "skills.log";
#ifdef PRINTF_TO_FILE
    remove(fileName);
    int stdDup = dup(1);
    FILE *outLog = fopen(fileName, "a");
    dup2(fileno(outLog), 1);
#endif

    printf("[文件] 所有printf的输出信息重定向到%s\n", fileName);

#ifdef PRINTF_TO_FILE
    fflush(stdout);
    fclose(outLog);
    dup2(stdDup, 1);
    close(stdDup);
#endif

    printf("[终端] 所有printf的输出信息恢复到终端\n");
}

int getRand(int a, int b)
{
    /*a,a+1,...,b-1,b*/
    return rand() % (b - a + 1) + a;
}

struct alignas(4) stc
{
    /*
    默认2个字节对齐,必须是2的倍数
    结构体指针和uint8_t指针的转换受到字节对齐和结构体内变量定义先后顺序的影响
    */
    unsigned int a : 4; // 占用4bit，来自字节对齐产生的多余空间
    unsigned int b : 4; // 占用4bit，来自字节对齐产生的多余空间
    uint8_t c;
    uint8_t d;
    uint8_t e;
    uint16_t f;
    uint16_t g;
};

extern void pause_continue();

void x86_64()
{
    uint32_t a = 1;
    uint8_t *p = (uint8_t *)&a;
    printf("__LITTLE_ENDIAN 地址的低位存储值的低位\n%u-%u-%u-%u\n", p[0], p[1], p[2], p[3]);
}

void progressBar()
{
    char str[100] = "Please be patient! We'll finish it in a minute! The mission is almost complete. We are happy!🚙";
    for (int i = 0; i <= 100; i++)
    {
        printf("\r");
        for (int j = 0; j < 100; j++)
        {
            if (j < i)
            {
                // printf("+");
                printf("%c", str[j]);
            }
            else
            {
                printf("-");
            }
        }
        printf("%d%%", i);
        fflush(stdout);
        usleep((rand() % (5 - 1 + 1) + 1) * pow(10, 4));
    }
    printf("\n");
}

void inputPassword()
{
    char *password = getpass("输入密码：");
    printf("%s\n", password);
    char a = 10;
    int b[3];
    auto A = a;
    auto &B = b;
    printf("%s\n", typeid(A).name());
    printf("%s\n", typeid(B).name());
}

float frand(int a, int b, int delta = 6)
{
    time_t t;
    assert(b >= a);
    b *= pow(10, delta);
    a *= pow(10, delta);
    float d = pow(10, -delta);
    srand((unsigned)time(&t));
    return (rand() % (b - a + 1) + a) * d;
}

void cpuPrefetch()
{
    float like = frand(0, 1);
    if (likely(like >= 0 && like <= 0.8))
    {
        printf("like=%f\n", like);
    }
    float unlike = frand(0, 1);
    if (unlikely(unlike >= 0.8 && unlike <= 1))
    {
        printf("unlike=%f\n", unlike);
    }
}

/*
联合类型的所有成员从同一内存地址开始
主机顺序__LITTLE_ENDIAN
地址的低位存储值的低位
修改任何一个成员都会导致其他成员的值发生变化
*/
union __zad__ Union
{
    char a;
    short b;
    int c;
};

/*
union和struct可以相互嵌套
且嵌套在内部的union或struct可以没有名称
*/
union un
{
    int a;
    union
    {
        int b;
        int c;
    };
};

struct st
{
    int a;
    struct
    {
        int b;
        int c;
    };
};

void learnUnion()
{
    Union u;
    printf("%ld\n", sizeof(Union));
    u.c = 52;
    printf("%d %d %d\n", u.a, u.b, u.c);
    u.c = 25;
    printf("%d %d %d\n", u.a, u.b, u.c);
}

enum string
{
    x1,
    x2,
    x3 = 10,
    x4,
    x5,
} x;

int main(int argc, char *argv[])
{
    strCat(5);
    // int number = 34;
    // printf("%p\n", &number);
    // redirectPrintf();
    char path[256];
    getWorkDir(path, sizeof(path), true);
    changeWorkDir(argv);
    // pause_continue();
    // x86_64();
    // inputPassword();
    // cpuPrefetch();
    // learnUnion();
    // int a(5), b(3);
    // printf("%d,%d,%d,%c,%c,%d\n", x, a, b, '\72', 72, ':');
    return 0;
}
/*
cd cpp;g++ -c -std=c++17 _skills/skills.cpp -o skills.o;g++ -g -std=c++17 skills.cpp -o skills skills.o;./skills;cd ..
cd cpp;rm -rf skills.o skills skills.log;cd ..
*/