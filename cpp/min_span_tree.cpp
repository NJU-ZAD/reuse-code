#include <bits/stdc++.h>
using namespace std;
#define v 9

int arr[v];

//*该函数返回与父节点距离最小的节点
int minIndex(int g[v][v], bool mstset[v], int dist[v])
{
	int min = INT_MAX, index;
	for (int i = 0; i < v; i++)
	{
		if (mstset[i] == false && dist[i] < min)
		{
			min = dist[i];
			index = i;
		}
	}
	return index;
}

void printMst(int dist[], int parent[])
{
	cout << "边"
		 << "\t"
		 << "权重" << endl;
	for (int counter = 1; counter < v; counter++)
	{
		cout << parent[counter] << " - " << counter << "\t" << dist[counter] << endl;
	}
}

//*Prim算法
void Prim(int g[v][v])
{
	cout << "Prim算法" << endl;
	int dist[v];
	int parent[v];
	bool mstset[v];
	for (int i = 0; i < v; i++)
	{
		mstset[i] = false;
		dist[i] = INT_MAX;
	}
	dist[0] = 0;
	//*根节点
	parent[0] = -1;
	for (int i = 0; i < v; i++)
	{
		int u = minIndex(g, mstset, dist);
		mstset[u] = true;
		//*循环更新所选节点的相邻节点的dist
		for (int j = 0; j < v; j++)
		{
			if (g[u][j] && mstset[j] == false && g[u][j] < dist[j])
			{
				dist[j] = g[u][j];
				parent[j] = u;
			}
		}
	}
	printMst(dist, parent);
}

bool cmp(pair<int, int> x, pair<int, int> y)
{
	return x.second < y.second;
}

int find(int x)
{
	if (arr[x] == -1)
		return x;
	return find(arr[x]);
}

bool mkMst(int x, int y)
{
	int a, b;
	a = find(x);
	b = find(y);
	if (a == x && b == y)
	{
		arr[x] = y;
		return true;
	}
	else if (a != b)
	{
		arr[a] = b;
		return true;
	}
	return false;
}

void printMst(vector<pair<int, int>> lst)
{
	cout << "size = " << lst.size() << endl;
	vector<pair<int, int>>::iterator itr;
	cout << "边"
		 << "\t"
		 << "权重" << endl;
	int i, j;
	for (itr = lst.begin(); itr != lst.end(); itr++)
	{
		if (itr->first % 10 == itr->first)
		{
			i = 0;
			j = itr->first;
			if (mkMst(i, j))
			{
				cout << i << " - " << j << "\t" << itr->second << endl;
			}
			else
			{
				continue;
			}
		}
		else
		{
			i = itr->first / 10;
			j = itr->first % 10;
			if (mkMst(i, j))
			{
				cout << i << " - " << j << "\t" << itr->second << endl;
			}
			else
			{
				continue;
			}
		}
	}
}

//*Kruskal算法
void Kruskal(int g[v][v])
{
	cout << "Kruskal算法" << endl;
	vector<pair<int, int>> set;
	map<int, int> lst;
	for (int i = 0; i < v; i++)
	{
		for (int j = i + 1; j < v; j++)
		{
			if (g[i][j])
			{
				lst.insert({10 * i + j, g[i][j]});
			}
		}
	}
	for (auto a : lst)
	{
		set.push_back(a);
	}
	sort(set.begin(), set.end(), cmp);
	printMst(set);
}

int main()
{
	int graph[v][v] = {{0, 4, 0, 0, 0, 0, 0, 8, 0},
					   {4, 0, 8, 0, 0, 0, 0, 11, 0},
					   {0, 8, 0, 7, 0, 4, 0, 0, 2},
					   {0, 0, 7, 0, 9, 14, 0, 0, 0},
					   {0, 0, 0, 9, 0, 10, 0, 0, 0},
					   {0, 0, 4, 14, 10, 0, 2, 0, 0},
					   {0, 0, 0, 0, 0, 2, 0, 1, 6},
					   {8, 11, 0, 0, 0, 0, 1, 0, 7},
					   {0, 0, 2, 0, 0, 0, 6, 7, 0}};
	Prim(graph);
	memset(arr, -1, sizeof(int) * v);
	Kruskal(graph);
}

/*
cd cpp;g++ -g -std=c++17 min_span_tree.cpp -o min_span_tree;./min_span_tree;cd ..
cd cpp;rm -rf min_span_tree;cd ..
*/