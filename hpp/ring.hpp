#pragma once
#include <iostream>
#include <thread>
#include <queue>
typedef unsigned int uint;
typedef unsigned long long ull;
class PacketRing
{
private:
	char *ringData; // 环数据流的内容
	ull charSize;	// 环数据流的大小
	// char *dataTag;	// 指示所有数据包尾部的逻辑位置
	std::queue<ull> dataTag; // 指示所有数据包尾部的逻辑位置
	ull writePos;			 // 指示当前插入的逻辑位置
	ull readPos;			 // 指示当前读出的逻辑位置
public:
	ull CurrReadNum;  // 指示当前已读数据包的个数
	ull CurrWriteNum; // 指示当前已写数据包的个数
	PacketRing(uint byteSize)
	{
		writePos = 0;
		readPos = 0;
		CurrReadNum = 0;
		CurrWriteNum = 0;
		charSize = byteSize / sizeof(char);
		if (charSize >= 1)
		{
			ringData = new char[charSize + 1];
			// dataTag = new char[charSize + 1];
			ringData[charSize] = '\0';
			// dataTag[charSize] = '\0';
		}
	}

	~PacketRing()
	{
		delete[] ringData;
		// delete[] dataTag;
		ringData = NULL;
		// dataTag = NULL;
	}

	bool IsCoverUnRead(uint length)
	{
		return (writePos + length - readPos) > charSize;
	}

	uint Push(char *buffData, uint length, bool safeMode)
	{
		if (buffData == NULL)
		{
			return -2;
		}
		if (length > charSize)
		{
			return -1;
		}
		if (safeMode && IsCoverUnRead(length))
		{
			return 1;
		}
		for (uint i = 0; i < length; i++)
		{
			ringData[GetArrayPos(GetArrayPos(writePos) + i)] = buffData[i];
		}
		writePos += length;
		dataTag.push(writePos);
		// printf("--------------------------Push PACKET----------------------------%lu Bytes\n", (length * sizeof(char)));
		CurrWriteNum++;
		return 0;
	}

	inline bool IsReadFinish(uint totalPacketNum)
	{
		return CurrReadNum == totalPacketNum;
	}

	bool Pop(char *&buffData, uint &length)
	{
		if (CurrReadNum >= CurrWriteNum)
		{
			return 1;
		}
		if (dataTag.empty())
		{
			return -1;
		}
		length = dataTag.front() - readPos;
		dataTag.pop();
		buffData = new char[length + 1];
		buffData[length] = '\0';
		for (uint i = 0; i < length; i++)
		{
			buffData[i] = ringData[GetArrayPos(GetArrayPos(readPos) + i)];
		}
		readPos += length;
		// printf("--------------------------Pop PACKET-----------------------------%s\n", buffData);
		CurrReadNum++;
		return 0;
	}

	inline uint GetArrayPos(uint pos)
	{
		return pos % charSize;
	}
};
