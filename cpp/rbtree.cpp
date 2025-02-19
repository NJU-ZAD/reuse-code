#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hpp/rbtree.hpp"

struct myNode
{
	struct rbNode node;
	char *str;
};

struct rbRoot mytree = (struct rbRoot){
	nullptr,
};

struct myNode *mySearch(struct rbRoot *root, char *str)
{
	struct rbNode *node = root->rbNode;

	while (node)
	{
		struct myNode *data = CONTAINER_OF(node, struct myNode, node);
		int result;

		result = strcmp(str, data->str);

		if (result < 0)
			node = node->rbLeft;
		else if (result > 0)
			node = node->rbRight;
		else
			return data;
	}
	return nullptr;
}

int myInsert(struct rbRoot *root, struct myNode *data)
{
	struct rbNode **nw = &(root->rbNode), *parent = nullptr;

	//*找出放置新节点的位置
	while (*nw)
	{
		struct myNode *ths = CONTAINER_OF(*nw, struct myNode, node);
		int result = strcmp(data->str, ths->str);

		parent = *nw;
		if (result < 0)
			nw = &((*nw)->rbLeft);
		else if (result > 0)
			nw = &((*nw)->rbRight);
		else
			return 0;
	}

	//*添加nw节点并重新平衡树
	rbLinkNode(&data->node, parent, nw);
	rbInsertColor(&data->node, root);

	return 1;
}

void myFree(struct myNode *node)
{
	if (node != nullptr)
	{
		if (node->str != nullptr)
		{
			free(node->str);
			node->str = nullptr;
		}
		free(node);
		node = nullptr;
	}
}

#define NUM_NODES 32

int main()
{
	struct myNode *mn[NUM_NODES];

	int i = 0;
	printf("插入节点1到NUM_NODES(32)：\n");
	for (; i < NUM_NODES; i++)
	{
		mn[i] = (struct myNode *)malloc(sizeof(struct myNode));
		mn[i]->str = (char *)malloc(sizeof(char) * 4);
		sprintf(mn[i]->str, "%d", i);
		myInsert(&mytree, mn[i]);
	}

	struct rbNode *node;
	printf("搜索所有节点：\n");
	for (node = rbFirst(&mytree); node; node = rbNext(node))
		printf("key = %s\n", RB_ENTRY(node, struct myNode, node)->str);

	printf("删除节点20：\n");
	struct myNode *data = mySearch(&mytree, (char *)"20");
	if (data)
	{
		rbErase(&data->node, &mytree);
		myFree(data);
	}

	printf("删除节点10：\n");
	data = mySearch(&mytree, (char *)"10");
	if (data)
	{
		rbErase(&data->node, &mytree);
		myFree(data);
	}

	printf("删除节点15：\n");
	data = mySearch(&mytree, (char *)"15");
	if (data)
	{
		rbErase(&data->node, &mytree);
		myFree(data);
	}

	printf("再次搜索：\n");
	for (node = rbFirst(&mytree); node; node = rbNext(node))
		printf("key = %s\n", RB_ENTRY(node, struct myNode, node)->str);
	return 0;
}

/*
cd cpp;g++ -c -std=c++17 rbtree.cpp -o rbtree.o;g++ -g -std=c++17 rbtree_tst.cpp rbtree.o -o rbtree_tst;./rbtree_tst;cd ..
cd cpp;rm -rf rbtree rbtree.o rbtree_tst;cd ..
*/
