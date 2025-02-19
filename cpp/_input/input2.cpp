#include <stdio.h>
#include <math.h>
#include <unistd.h>

void car(int distance)
{
    for (int i = 0; i <= distance; i++)
    {
        printf("\r");
        printf("🏁");
        for (int j = 0; j < distance - i; j++)
        {
            printf("_");
        }
        // printf("🚙");
        printf("🚒");
        // printf("🏍");
        for (int j = 0; j <= i; j++)
        {
            printf("_");
        }
        printf("🚩");
        fflush(stdout);
        usleep((15 - i / 10.0 + 1) * pow(10, 4));
        // usleep((rand() % (8 - 4 + 1) + 4) * pow(10, 4));
    }

    printf("\n");
}