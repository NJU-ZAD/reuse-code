#pragma once
extern void getWorkDir(char *path, int len, bool print = false);
extern void getMainDir(char *path, char *argv[], bool print = false);
extern void changeWorkDir(char *argv[]);