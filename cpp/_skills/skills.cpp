#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

static char workDir[256];
static char fileDir[256];

// 获取当前工作目录
void getWorkDir(char *path, int len, bool print = false)
{
    getcwd(path, len);
    if (print == true)
        printf("当前工作目录为%s\n", path);
}

// 获取主函数目录
// 当主函数所在源文件与编译生成的可执行文件位于相同目录时该函数有效
void getMainDir(char *path, char *argv[], bool print = false)
{
    int len = strlen(argv[0]);
    memcpy(path, argv[0], len);
    path[len] = '\0';
    if (path[0] == '.')
    {
        char dir[256];
        getWorkDir(dir, sizeof(dir));
        int l = strlen(dir);
        memcpy(dir + l, path + 1, len);
        len += l;
        memcpy(path, dir, len);
    }
    for (int i = len - 1; i >= 0; i--)
    {
        if (path[i] == '/')
        {
            path[i] = '\0';
            break;
        }
    }
    if (print == true)
        printf("主函数目录为%s\n", path);
}

// 将工作目录切换为主函数目录
void changeWorkDir(char *argv[])
{
    char path[256];
    getMainDir(path, argv);
    chdir(path);
    printf("将工作目录切换为主函数目录\n");
    getWorkDir(path, sizeof(path), true);
}

void pause_continue()
{
    printf("按回车键继续...\n");
    system("read REPLY");
}
