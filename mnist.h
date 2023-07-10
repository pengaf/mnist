#pragma once
//http://yann.lecun.com/exdb/mnist/

#include <fstream>
#include <vector>
#include <string>

const uint32_t mnist_image_header_flag = 0x00000803;
const uint32_t mnist_label_header_flag = 0x00000801;

struct MnistImageHeader
{
	uint32_t magicNumber;
	uint32_t imageCount;
	uint32_t rowCount;
	uint32_t columnCount;
};

struct MnistLabelHeader
{
	uint32_t magicNumber;
	uint32_t labelCount;
};

uint32_t ConvertEndian(uint32_t n)
{
	return ((n << 24) & 0xFF000000) | ((n << 8) & 0x00FF0000) | ((n >> 8) & 0x0000FF00) | ((n >> 24) & 0x000000FF);
}

inline bool ReadImageData(MnistImageHeader& header, std::vector<uint8_t>& data, const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}
	uint32_t tmp;
	file.read((char*)&tmp, sizeof(uint32_t));
	header.magicNumber = ConvertEndian(tmp);
	if (header.magicNumber != mnist_image_header_flag)
	{
		return false;
	}
	file.read((char*)&tmp, sizeof(uint32_t));
	header.imageCount = ConvertEndian(tmp);
	file.read((char*)&tmp, sizeof(uint32_t));
	header.rowCount = ConvertEndian(tmp);
	file.read((char*)&tmp, sizeof(uint32_t));
	header.columnCount = ConvertEndian(tmp);

	uint32_t dataSize = header.imageCount*header.rowCount*header.columnCount;
	data.resize(dataSize);
	file.read((char*)data.data(), dataSize);
	return true;
}

inline bool ReadLabelData(MnistLabelHeader& header, std::vector<uint8_t>& data, const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}
	uint32_t tmp;
	file.read((char*)&tmp, sizeof(uint32_t));
	header.magicNumber = ConvertEndian(tmp);
	if (header.magicNumber != mnist_label_header_flag)
	{
		return false;
	}
	file.read((char*)&tmp, sizeof(uint32_t));
	header.labelCount = ConvertEndian(tmp);

	uint32_t dataSize = header.labelCount;
	data.resize(dataSize);
	file.read((char*)data.data(), dataSize);
	return true;
}
