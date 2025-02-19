#include "../hpp/ring.hpp"

char *tempPacket = NULL;
uint tempPacketSize;
PacketRing pr(500 * 1024 * 1024); // 分配500MB的环空间
uint packetNum = 500000;		  // 数据包的个数

void Flow()
{
	char *packet = NULL;
	uint min = 64 / sizeof(char);
	uint max = 1513 / sizeof(char);
	for (uint index = 0; index < packetNum; index++)
	{
		// 限制数据包的大小在64字节到1513字节
		uint size = rand() % (max - min) + min;
		// 随机生成单个数据包
		packet = new char[size + 1];
		packet[size] = '\0';
		for (uint i = 0; i < size; i++)
		{
			// 可显示字符的ASCII在32和126之间
			packet[i] = rand() % (126 - 32) + 32;
		}
		if (pr.Push(packet, size, true) == 0)
		{
			delete[] packet;
			packet = NULL;
		}
	}
}

void Read()
{
	while (!pr.IsReadFinish(packetNum))
	{
		if (pr.Pop(tempPacket, tempPacketSize) == 0)
		{
			delete[] tempPacket;
			tempPacket = NULL;
		}
	}
}

void Display()
{
	int end = 0;
	while (1)
	{
		if (end == 3)
			break;
		if (pr.CurrReadNum <= packetNum)
		{
			fflush(stdout);
			printf("write:%llu---read:%llu---interval:%llu\r", pr.CurrWriteNum, pr.CurrReadNum, pr.CurrWriteNum - pr.CurrReadNum);
			if (pr.CurrReadNum == packetNum)
			{
				end++;
			}
		}
	}
}

int main()
{
	std::thread threadFlow(Flow);		// 创建生成数据包的线程
	std::thread threadRead(Read);		// 创建读出数据包的线程
	std::thread threadDisplay(Display); // 创建显示的线程
	threadFlow.join();
	threadRead.join();
	threadDisplay.join();
}
/*
cd cpp;g++ -g -std=c++17 netflow.cpp -o netflow;./netflow;cd ..
cd cpp;rm -rf netflow;cd ..
*/