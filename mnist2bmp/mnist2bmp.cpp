#include "../mnist.h"
#include <windows.h>
#include <pshpack2.h>

struct BmpFileHeader24 : BITMAPFILEHEADER, BITMAPINFOHEADER
{
public:
	BmpFileHeader24(uint32_t width, uint32_t height)
	{
		uint32_t bytesPerPixel = 3;
		uint32_t bytesPerRow = (width * bytesPerPixel + 3) / 4 * 4;
		uint32_t pixelBytes = bytesPerRow * height;

		this->bfType = 0x4D42;
		this->bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + pixelBytes;
		this->bfReserved1 = 0;
		this->bfReserved2 = 0;
		this->bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

		this->biSize = sizeof(BITMAPINFOHEADER);
		this->biWidth = width;
		this->biHeight = -long(height);
		this->biPlanes = 1;
		this->biBitCount = bytesPerPixel * 8;
		this->biCompression = BI_RGB;
		this->biSizeImage = pixelBytes;
		this->biXPelsPerMeter = 0;
		this->biYPelsPerMeter = 0;
		this->biClrUsed = 0;
		this->biClrImportant = 0;
	}

	uint32_t getBytesPerRow() const
	{
		uint32_t bytesPerPixel = this->biBitCount / 8;
		uint32_t bytesPerRow = (biWidth * bytesPerPixel + 3) / 4 * 4;
		return bytesPerRow;
	}
};

struct BmpFileHeader8 : BITMAPFILEHEADER, BITMAPINFOHEADER
{
	RGBQUAD palette[256];
public:
	BmpFileHeader8(uint32_t width, uint32_t height)
	{
		uint32_t bytesPerPixel = 1;
		uint32_t bytesPerRow = (width * bytesPerPixel + 3) / 4 * 4;
		uint32_t pixelBytes = bytesPerRow * height;

		this->bfType = 0x4D42;
		this->bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(palette) + pixelBytes;
		this->bfReserved1 = 0;
		this->bfReserved2 = 0;
		this->bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(palette);

		this->biSize = sizeof(BITMAPINFOHEADER);
		this->biWidth = width;
		this->biHeight = -long(height);
		this->biPlanes = 1;
		this->biBitCount = bytesPerPixel * 8;
		this->biCompression = BI_RGB;
		this->biSizeImage = pixelBytes;
		this->biXPelsPerMeter = 0;
		this->biYPelsPerMeter = 0;
		this->biClrUsed = 0;
		this->biClrImportant = 0;

		for (uint32_t i = 0; i < 256; ++i)
		{
			RGBQUAD& rgb = this->palette[i];
			BYTE gray = i;
			rgb.rgbBlue = gray;
			rgb.rgbGreen = gray;
			rgb.rgbRed = gray;
			rgb.rgbReserved = 0;
		}
	}

	uint32_t getBytesPerRow() const
	{
		uint32_t bytesPerPixel = this->biBitCount / 8;
		uint32_t bytesPerRow = (biWidth * bytesPerPixel + 3) / 4 * 4;
		return bytesPerRow;
	}
};

#include <poppack.h>

void mnist2bmp(const std::string& bmpFileName, const std::string&  mnistFileName, uint32_t imagePerRow = 100)
{
	MnistImageHeader mnistImageHeader;
	std::ifstream mnistFile(mnistFileName, std::ios::binary);
	if (!mnistFile.is_open())
	{
		return;
	}
	uint32_t tmp;
	mnistFile.read((char*)&tmp, sizeof(uint32_t));
	mnistImageHeader.magicNumber = ConvertEndian(tmp);
	if (mnistImageHeader.magicNumber != 0x00000803)
	{
		return;
	}
	mnistFile.read((char*)&tmp, sizeof(uint32_t));
	mnistImageHeader.imageCount = ConvertEndian(tmp);
	mnistFile.read((char*)&tmp, sizeof(uint32_t));
	mnistImageHeader.rowCount = ConvertEndian(tmp);
	mnistFile.read((char*)&tmp, sizeof(uint32_t));
	mnistImageHeader.columnCount = ConvertEndian(tmp);

	if (imagePerRow < 1)
	{
		imagePerRow = 1;
	}
	uint32_t imageRows = (mnistImageHeader.imageCount + imagePerRow - 1) / imagePerRow;
	uint32_t bmpWidth = mnistImageHeader.columnCount * imagePerRow;
	uint32_t bmpHeight = mnistImageHeader.rowCount * imageRows;

	BmpFileHeader8 bmpFileHeader(bmpWidth, bmpHeight);
	
	std::ofstream bmpFile(bmpFileName, std::ios::binary);
	bmpFile.write((char*)&bmpFileHeader, sizeof(bmpFileHeader));

	uint32_t bmpBytesPerRow = bmpFileHeader.getBytesPerRow();

	uint32_t mnistBufferSize = mnistImageHeader.columnCount * mnistImageHeader.rowCount;
	char* mnistBuffer = new char[mnistBufferSize];
	uint32_t bmpBufferSize = bmpBytesPerRow * mnistImageHeader.rowCount;
	char* bmpBuffer = new char[bmpBufferSize];

	for (uint32_t i = 0; i < imageRows; ++i)
	{
		memset(bmpBuffer, 0, bmpBufferSize);
		for (uint32_t j = 0; j < imagePerRow; ++j)
		{
			if (i * imagePerRow + j > mnistImageHeader.imageCount)
			{
				break;
			}
			mnistFile.read(mnistBuffer, mnistBufferSize);
			for (uint32_t k = 0; k < mnistImageHeader.rowCount; ++k)
			{
				memcpy(&bmpBuffer[k * bmpBytesPerRow + j * mnistImageHeader.columnCount], &mnistBuffer[k * mnistImageHeader.columnCount], mnistImageHeader.columnCount);
			}
		}
		bmpFile.write(bmpBuffer, bmpBufferSize);
	}
}

int main()
{
	std::string path = CMAKE_SOURCE_DIR;
	mnist2bmp(path + "/data/train-images.bmp", path + "/data/train-images.idx3-ubyte");
	mnist2bmp(path + "/data/t10k-images.bmp", path + "/data/t10k-images.idx3-ubyte");
}
